use std::collections::HashMap;
use std::io::Write;
use std::process::Stdio;
use std::time::{
    Duration,
    Instant,
};

use bstr::ByteSlice;
use clap::Args;
use crossterm::style::{
    self,
    Attribute,
    Color,
    Stylize,
};
use crossterm::{
    cursor,
    execute,
    queue,
    terminal,
};
use eyre::{
    Result,
    eyre,
};
use futures::stream::{
    FuturesUnordered,
    StreamExt,
};
use spinners::{
    Spinner,
    Spinners,
};

use crate::cli::agent::hook::{
    Hook,
    HookTrigger,
};
use crate::cli::chat::consts::AGENT_FORMAT_HOOKS_DOC_URL;
use crate::cli::chat::util::truncate_safe;
use crate::cli::chat::{
    ChatError,
    ChatSession,
    ChatState,
};

#[derive(Debug, Clone)]
pub struct CachedHook {
    output: String,
    expiry: Option<Instant>,
}

/// Maps a hook name to a [`CachedHook`]
#[derive(Debug, Clone, Default)]
pub struct HookExecutor {
    pub cache: HashMap<(HookTrigger, Hook), CachedHook>,
}

impl HookExecutor {
    pub fn new() -> Self {
        Self { cache: HashMap::new() }
    }

    /// Run and cache [`Hook`]s. Any hooks that are already cached will be returned without
    /// executing. Hooks that fail to execute will not be returned. Returned hook order is
    /// undefined.
    ///
    /// If `updates` is `Some`, progress on hook execution will be written to it.
    /// Errors encountered with write operations to `updates` are ignored.
    ///
    /// Note: [`HookTrigger::AgentSpawn`] hooks never leave the cache.
    pub async fn run_hooks(
        &mut self,
        hooks: HashMap<HookTrigger, Vec<Hook>>,
        output: &mut impl Write,
        prompt: Option<&str>,
    ) -> Result<Vec<((HookTrigger, Hook), String)>, ChatError> {
        let mut cached = vec![];
        let mut futures = FuturesUnordered::new();
        for hook in hooks
            .into_iter()
            .flat_map(|(trigger, hooks)| hooks.into_iter().map(move |hook| (trigger, hook)))
        {
            if let Some(cache) = self.get_cache(&hook) {
                cached.push((hook.clone(), cache.clone()));
                continue;
            }
            futures.push(self.run_hook(hook, prompt));
        }

        let mut complete = 0;
        let total = futures.len();
        let mut spinner = None;
        let spinner_text = |complete: usize, total: usize| {
            format!(
                "{} of {} hooks finished",
                complete.to_string().blue(),
                total.to_string().blue(),
            )
        };

        if total != 0 {
            spinner = Some(Spinner::new(Spinners::Dots12, spinner_text(complete, total)));
        }

        // Process results as they complete
        let mut results = vec![];
        let start_time = Instant::now();
        while let Some((hook, result, duration)) = futures.next().await {
            // If output is enabled, handle that first
            if let Some(spinner) = spinner.as_mut() {
                spinner.stop();

                // Erase the spinner
                execute!(
                    output,
                    cursor::MoveToColumn(0),
                    terminal::Clear(terminal::ClearType::CurrentLine),
                    cursor::Hide,
                )?;
            }

            if let Err(err) = &result {
                let hook_desc = if let Some(tool_name) = &hook.1.tool_name {
                    format!("tool:{}", tool_name)
                } else if let Some(command) = &hook.1.command {
                    command.clone()
                } else {
                    "unknown hook".to_string()
                };
                
                queue!(
                    output,
                    style::SetForegroundColor(style::Color::Red),
                    style::Print("✗ "),
                    style::SetForegroundColor(style::Color::Blue),
                    style::Print(&hook_desc),
                    style::ResetColor,
                    style::Print(" failed after "),
                    style::SetForegroundColor(style::Color::Yellow),
                    style::Print(format!("{:.2} s", duration.as_secs_f32())),
                    style::ResetColor,
                    style::Print(format!(": {}\n", err)),
                )?;
            }

            // Process results regardless of output enabled
            if let Ok(output) = result {
                complete += 1;
                results.push((hook, output));
            }

            // Display ending summary or add a new spinner
            // The futures set size decreases each time we process one
            if futures.is_empty() {
                let symbol = if total == complete {
                    "✓".to_string().green()
                } else {
                    "✗".to_string().red()
                };

                queue!(
                    output,
                    style::SetForegroundColor(Color::Blue),
                    style::Print(format!("{symbol} {} in ", spinner_text(complete, total))),
                    style::SetForegroundColor(style::Color::Yellow),
                    style::Print(format!("{:.2} s\n", start_time.elapsed().as_secs_f32())),
                    style::ResetColor,
                )?;
            } else {
                spinner = Some(Spinner::new(Spinners::Dots, spinner_text(complete, total)));
            }
        }
        drop(futures);

        // Fill cache with executed results, skipping what was already from cache
        for ((trigger, hook), output) in &results {
            self.cache.insert((*trigger, hook.clone()), CachedHook {
                output: output.clone(),
                expiry: match trigger {
                    HookTrigger::AgentSpawn => None,
                    HookTrigger::UserPromptSubmit => Some(Instant::now() + Duration::from_secs(hook.cache_ttl_seconds)),
                },
            });
        }

        results.append(&mut cached);

        Ok(results)
    }

    async fn run_hook(
        &self,
        hook: (HookTrigger, Hook),
        prompt: Option<&str>,
    ) -> ((HookTrigger, Hook), Result<String>, Duration) {
        let start_time = Instant::now();

        let result = if let Some(tool_name) = &hook.1.tool_name {
            // Execute tool
            self.execute_tool(tool_name, &hook.1.tool_args, prompt).await
        } else if let Some(command) = &hook.1.command {
            // Execute shell command
            self.execute_command(command, &hook.1, prompt).await
        } else {
            Err(eyre!("Hook must have either command or tool_name"))
        };

        (hook, result, start_time.elapsed())
    }

    async fn execute_tool(
        &self,
        tool_name: &str,
        tool_args: &Option<serde_json::Value>,
        prompt: Option<&str>,
    ) -> Result<String> {
        // Replace ${USER_PROMPT} in tool_args if prompt is provided
        let mut substituted_args = tool_args.clone();
        if let (Some(args), Some(user_prompt)) = (&mut substituted_args, prompt) {
            if let Some(obj) = args.as_object_mut() {
                for (_, value) in obj {
                    if let Some(s) = value.as_str() {
                        *value = serde_json::Value::String(s.replace("${USER_PROMPT}", user_prompt));
                    }
                }
            }
        }
        use crate::cli::chat::tools::Tool;
        use crate::os::Os;
        use std::collections::HashMap;
        
        // Create OS instance for tool execution
        let os = Os::new().await?;
        
        // Parse tool arguments
        let default_args = serde_json::Value::Object(serde_json::Map::new());
        let args = substituted_args.as_ref().unwrap_or(&default_args);
        
        // Create tool instance based on name and arguments
        let tool = match tool_name {
            "resource" => {
                let resource_tool: crate::cli::chat::tools::resource::Resource = 
                    serde_json::from_value(args.clone())?;
                Tool::Resource(resource_tool)
            },
            "fs_read" => {
                let fs_read_tool: crate::cli::chat::tools::fs_read::FsRead = 
                    serde_json::from_value(args.clone())?;
                Tool::FsRead(fs_read_tool)
            },
            "fs_write" => {
                let fs_write_tool: crate::cli::chat::tools::fs_write::FsWrite = 
                    serde_json::from_value(args.clone())?;
                Tool::FsWrite(fs_write_tool)
            },
            "execute_bash" | "execute_cmd" => {
                let execute_tool: crate::cli::chat::tools::execute::ExecuteCommand = 
                    serde_json::from_value(args.clone())?;
                Tool::ExecuteCommand(execute_tool)
            },
            "use_aws" => {
                let aws_tool: crate::cli::chat::tools::use_aws::UseAws = 
                    serde_json::from_value(args.clone())?;
                Tool::UseAws(aws_tool)
            },
            _ => return Err(eyre!("Unsupported tool: {}", tool_name)),
        };
        
        // Execute the tool
        let mut output = Vec::new();
        let mut line_tracker = HashMap::new();
        let invoke_result = tool.invoke(&os, &mut output, &mut line_tracker, None).await?;
        
        let result = invoke_result.as_str().to_string();
        
        // Return the tool result as string
        Ok(result)
    }

    async fn execute_command(
        &self,
        command: &str,
        hook: &Hook,
        prompt: Option<&str>,
    ) -> Result<String> {
        #[cfg(unix)]
        let mut cmd = tokio::process::Command::new("bash");
        #[cfg(unix)]
        let cmd = cmd
            .arg("-c")
            .arg(command)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        #[cfg(windows)]
        let mut cmd = tokio::process::Command::new("cmd");
        #[cfg(windows)]
        let cmd = cmd
            .arg("/C")
            .arg(command)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let timeout = Duration::from_millis(hook.timeout_ms);

        // Set USER_PROMPT environment variable if provided
        if let Some(prompt) = prompt {
            // Sanitize the prompt to avoid issues with special characters
            let sanitized_prompt = sanitize_user_prompt(prompt);
            cmd.env("USER_PROMPT", sanitized_prompt);
        }

        let command_future = cmd.output();

        // Run with timeout
        match tokio::time::timeout(timeout, command_future).await {
            Ok(Ok(result)) => {
                if result.status.success() {
                    let stdout = result.stdout.to_str_lossy();
                    let stdout = format!(
                        "{}{}",
                        truncate_safe(&stdout, hook.max_output_size),
                        if stdout.len() > hook.max_output_size {
                            " ... truncated"
                        } else {
                            ""
                        }
                    );
                    Ok(stdout)
                } else {
                    Err(eyre!("command returned non-zero exit code: {}", result.status))
                }
            },
            Ok(Err(err)) => Err(eyre!("failed to execute command: {}", err)),
            Err(_) => Err(eyre!("command timed out after {} ms", timeout.as_millis())),
        }
    }

    /// Will return a cached hook's output if it exists and isn't expired.
    fn get_cache(&self, hook: &(HookTrigger, Hook)) -> Option<String> {
        self.cache.get(hook).and_then(|o| {
            if let Some(expiry) = o.expiry {
                if Instant::now() < expiry {
                    Some(o.output.clone())
                } else {
                    None
                }
            } else {
                Some(o.output.clone())
            }
        })
    }
}

/// Sanitizes a string value to be used as an environment variable
fn sanitize_user_prompt(input: &str) -> String {
    // Limit the size of input to first 4096 characters
    let truncated = if input.len() > 4096 { &input[0..4096] } else { input };

    // Remove any potentially problematic characters
    truncated.replace(|c: char| c.is_control() && c != '\n' && c != '\r' && c != '\t', "")
}

#[deny(missing_docs)]
#[derive(Debug, PartialEq, Args)]
#[command(
    before_long_help = "Use context hooks to specify shell commands to run. The output from these 
commands will be appended to the prompt to Amazon Q.

Refer to the documentation for how to configure hooks with your agent: https://github.com/aws/amazon-q-developer-cli/blob/main/docs/agent-format.md#hooks-field

Notes:
• Hooks are executed in parallel
• 'conversation_start' hooks run on the first user prompt and are attached once to the conversation history sent to Amazon Q
• 'per_prompt' hooks run on each user prompt and are attached to the prompt, but are not stored in conversation history"
)]
pub struct HooksArgs;

impl HooksArgs {
    pub async fn execute(self, session: &mut ChatSession) -> Result<ChatState, ChatError> {
        let Some(context_manager) = &mut session.conversation.context_manager else {
            return Ok(ChatState::PromptUser {
                skip_printing_tools: true,
            });
        };

        let mut out = Vec::new();
        for (trigger, hooks) in &context_manager.hooks {
            writeln!(&mut out, "{trigger}:")?;
            match hooks.is_empty() {
                true => writeln!(&mut out, "<none>")?,
                false => {
                    for hook in hooks {
                        let hook_desc = if let Some(tool_name) = &hook.tool_name {
                            format!("tool: {}", tool_name)
                        } else if let Some(command) = &hook.command {
                            command.clone()
                        } else {
                            "unknown hook".to_string()
                        };
                        writeln!(&mut out, "  - {}", hook_desc)?;
                    }
                },
            }
        }

        if out.is_empty() {
            queue!(
                session.stderr,
                style::Print(
                    "No hooks are configured.\n\nRefer to the documentation for how to add hooks to your agent: "
                ),
                style::SetForegroundColor(Color::Green),
                style::Print(AGENT_FORMAT_HOOKS_DOC_URL),
                style::SetAttribute(Attribute::Reset),
                style::Print("\n"),
            )?;
        } else {
            session.stdout.write_all(&out)?;
        }

        Ok(ChatState::PromptUser {
            skip_printing_tools: true,
        })
    }
}
