// Code generated by software.amazon.smithy.rust.codegen.smithy-rs. DO NOT EDIT.
pub(crate) fn de_interaction_component<'a, I>(
    tokens: &mut ::std::iter::Peekable<I>,
) -> ::std::result::Result<
    Option<crate::types::InteractionComponent>,
    ::aws_smithy_json::deserialize::error::DeserializeError,
>
where
    I: Iterator<
        Item = Result<
            ::aws_smithy_json::deserialize::Token<'a>,
            ::aws_smithy_json::deserialize::error::DeserializeError,
        >,
    >,
{
    match tokens.next().transpose()? {
        Some(::aws_smithy_json::deserialize::Token::ValueNull { .. }) => Ok(None),
        Some(::aws_smithy_json::deserialize::Token::StartObject { .. }) => {
            #[allow(unused_mut)]
            let mut builder = crate::types::builders::InteractionComponentBuilder::default();
            loop {
                match tokens.next().transpose()? {
                    Some(::aws_smithy_json::deserialize::Token::EndObject { .. }) => break,
                    Some(::aws_smithy_json::deserialize::Token::ObjectKey { key, .. }) => match key
                        .to_unescaped()?
                        .as_ref()
                    {
                        "text" => {
                            builder = builder.set_text(crate::protocol_serde::shape_text::de_text(tokens)?);
                        },
                        "alert" => {
                            builder = builder.set_alert(crate::protocol_serde::shape_alert::de_alert(tokens)?);
                        },
                        "infrastructureUpdate" => {
                            builder = builder.set_infrastructure_update(
                                crate::protocol_serde::shape_infrastructure_update::de_infrastructure_update(tokens)?,
                            );
                        },
                        "progress" => {
                            builder = builder.set_progress(crate::protocol_serde::shape_progress::de_progress(tokens)?);
                        },
                        "step" => {
                            builder = builder.set_step(crate::protocol_serde::shape_step::de_step(tokens)?);
                        },
                        "taskDetails" => {
                            builder = builder
                                .set_task_details(crate::protocol_serde::shape_task_details::de_task_details(tokens)?);
                        },
                        "taskReference" => {
                            builder = builder.set_task_reference(
                                crate::protocol_serde::shape_task_reference::de_task_reference(tokens)?,
                            );
                        },
                        "suggestions" => {
                            builder = builder
                                .set_suggestions(crate::protocol_serde::shape_suggestions::de_suggestions(tokens)?);
                        },
                        "section" => {
                            builder = builder.set_section(crate::protocol_serde::shape_section::de_section(tokens)?);
                        },
                        "resource" => {
                            builder = builder.set_resource(crate::protocol_serde::shape_resource::de_resource(tokens)?);
                        },
                        "resourceList" => {
                            builder = builder.set_resource_list(
                                crate::protocol_serde::shape_resource_list::de_resource_list(tokens)?,
                            );
                        },
                        "action" => {
                            builder = builder.set_action(crate::protocol_serde::shape_action::de_action(tokens)?);
                        },
                        _ => ::aws_smithy_json::deserialize::token::skip_value(tokens)?,
                    },
                    other => {
                        return Err(::aws_smithy_json::deserialize::error::DeserializeError::custom(
                            format!("expected object key or end object, found: {:?}", other),
                        ));
                    },
                }
            }
            Ok(Some(builder.build()))
        },
        _ => Err(::aws_smithy_json::deserialize::error::DeserializeError::custom(
            "expected start object or null",
        )),
    }
}
