#!/usr/bin/env python3

import os
import platform
import pathlib
import shutil
import subprocess
from enum import Enum

# Constants
APP_NAME = "Amazon Q"
CLI_BINARY_NAME = "q"
PTY_BINARY_NAME = "qterm"
CLI_PACKAGE_NAME = "q_cli"
PTY_PACKAGE_NAME = "figterm"
BUILD_DIR = pathlib.Path("build").absolute()

def run_cmd(args, env=None, cwd=None):
    print(f"+ {' '.join(str(arg) for arg in args)}")
    subprocess.run(args, env=env, cwd=cwd, check=True)

def is_darwin():
    return platform.system() == "Darwin"

def get_target_triple():
    if is_darwin():
        return "universal-apple-darwin"
    else:
        return f"{platform.machine()}-unknown-linux-gnu"

def rust_targets():
    if is_darwin():
        return ["x86_64-apple-darwin", "aarch64-apple-darwin"]
    else:
        return [get_target_triple()]

def rust_env(release=True):
    env = {
        "CARGO_NET_GIT_FETCH_WITH_CLI": "true",
    }
    
    if release:
        rustflags = ["-C force-frame-pointers=yes"]
            
        env["CARGO_INCREMENTAL"] = "0"
        env["CARGO_PROFILE_RELEASE_LTO"] = "thin"
        env["RUSTFLAGS"] = " ".join(rustflags)
    
    if is_darwin():
        env["MACOSX_DEPLOYMENT_TARGET"] = "10.13"
    
    env["AMAZON_Q_BUILD_TARGET_TRIPLE"] = get_target_triple()
    
    return env

def build_cargo_bin(release, package, output_name=None, targets=None):
    if targets is None:
        targets = rust_targets()
        
    args = ["cargo", "build", "--locked", "--package", package]
    
    if release:
        args.append("--release")
    
    for target in targets:
        args.extend(["--target", target])
    
    run_cmd(
        args,
        env={
            **os.environ,
            **rust_env(release=release),
        },
    )
    
    if release:
        target_subdir = "release"
    else:
        target_subdir = "debug"
    
    # Create "universal" binary for macOS
    if is_darwin():
        out_path = BUILD_DIR / f"{output_name or package}-universal-apple-darwin"
        args = [
            "lipo",
            "-create",
            "-output",
            out_path,
        ]
        for target in targets:
            args.append(pathlib.Path("target") / target / target_subdir / package)
        run_cmd(args)
        return out_path
    else:
        # Linux does not cross compile arch
        target = targets[0]
        target_path = pathlib.Path("target") / target / target_subdir / package
        out_path = BUILD_DIR / "bin" / f"{(output_name or package)}-{target}"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target_path, out_path)
        return out_path

def build_macos_dmg(cli_path, pty_path):
    """Creates a minimal macOS DMG with the CLI and PTY binaries"""
    print(f"Building macOS DMG with:\n- CLI: {cli_path}\n- PTY: {pty_path}")
    
    # Create app structure
    app_path = BUILD_DIR / f"{APP_NAME}.app"
    shutil.rmtree(app_path, ignore_errors=True)
    
    # Create basic app structure
    (app_path / "Contents/MacOS").mkdir(parents=True, exist_ok=True)
    (app_path / "Contents/Resources").mkdir(parents=True, exist_ok=True)
    
    # Copy binaries
    shutil.copy2(cli_path, app_path / f"Contents/MacOS/{CLI_BINARY_NAME}")
    shutil.copy2(pty_path, app_path / f"Contents/MacOS/{PTY_BINARY_NAME}")
    
    # Create Info.plist
    info_plist = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDisplayName</key>
    <string>Amazon Q</string>
    <key>CFBundleExecutable</key>
    <string>q</string>
    <key>CFBundleIdentifier</key>
    <string>com.amazon.q</string>
    <key>CFBundleName</key>
    <string>Amazon Q</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleVersion</key>
    <string>1.0</string>
    <key>LSUIElement</key>
    <true/>
</dict>
</plist>
"""
    (app_path / "Contents/Info.plist").write_text(info_plist)
    
    # Create DMG
    dmg_path = BUILD_DIR / f"{APP_NAME}.dmg"
    dmg_path.unlink(missing_ok=True)
    
    try:
        import dmgbuild
        print("Using dmgbuild to create DMG...")
        dmgbuild.build_dmg(
            volume_name=APP_NAME,
            filename=dmg_path,
            settings={
                "format": "ULFO",
                "files": [str(app_path)],
                "symlinks": {"Applications": "/Applications"},
                "icon_size": 160,
                "window_rect": ((100, 100), (660, 400)),
            },
        )
        print(f"Created DMG at {dmg_path}")
    except ImportError:
        print("dmgbuild not installed. Using hdiutil instead...")
        run_cmd(["hdiutil", "create", "-volname", APP_NAME, 
                 "-srcfolder", app_path, "-ov", "-format", "UDZO", dmg_path])
        print(f"Created DMG at {dmg_path}")
    
    return dmg_path

def main():
    # Create build directory
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    
    # Build binaries
    print(f"Building {CLI_PACKAGE_NAME}")
    cli_path = build_cargo_bin(
        release=True,
        package=CLI_PACKAGE_NAME,
        output_name=CLI_BINARY_NAME,
    )
    
    print(f"Building {PTY_PACKAGE_NAME}")
    pty_path = build_cargo_bin(
        release=True,
        package=PTY_PACKAGE_NAME,
        output_name=PTY_BINARY_NAME,
    )
    
    # Build macOS DMG
    print("\nBuilding for macOS...")
    dmg_path = build_macos_dmg(cli_path, pty_path)
    print(f"\nMacOS DMG created at: {dmg_path}")
    
    print("\nBuild completed successfully!")

if __name__ == "__main__":
    main()
