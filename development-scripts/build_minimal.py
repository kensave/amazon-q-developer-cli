#!/usr/bin/env python3

import argparse
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
DESKTOP_BINARY_NAME = "q-desktop"
CLI_PACKAGE_NAME = "q_cli"
PTY_PACKAGE_NAME = "figterm"
DESKTOP_PACKAGE_NAME = "fig_desktop"
DMG_NAME = APP_NAME
LINUX_PACKAGE_NAME = "amazon-q"
LINUX_ARCHIVE_NAME = "q"

BUILD_DIR = pathlib.Path("build").absolute()

class Variant(Enum):
    FULL = 1
    MINIMAL = 2

def run_cmd(args, env=None, cwd=None):
    print(f"+ {' '.join(str(arg) for arg in args)}")
    subprocess.run(args, env=env, cwd=cwd, check=True)

def run_cmd_output(args, env=None, cwd=None):
    res = subprocess.run(args, env=env, cwd=cwd, check=True, stdout=subprocess.PIPE)
    return res.stdout.decode("utf-8")

def is_darwin():
    return platform.system() == "Darwin"

def is_linux():
    return platform.system() == "Linux"

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

def rust_env(release=True, variant=None):
    env = {
        "CARGO_NET_GIT_FETCH_WITH_CLI": "true",
    }
    
    if release:
        rustflags = ["-C force-frame-pointers=yes"]
        
        if is_linux():
            rustflags.append("-C link-arg=-Wl,--compress-debug-sections=zlib")
            
        env["CARGO_INCREMENTAL"] = "0"
        env["CARGO_PROFILE_RELEASE_LTO"] = "thin"
        env["RUSTFLAGS"] = " ".join(rustflags)
    
    if is_darwin():
        env["MACOSX_DEPLOYMENT_TARGET"] = "10.13"
    
    env["AMAZON_Q_BUILD_TARGET_TRIPLE"] = get_target_triple()
    if variant:
        env["AMAZON_Q_BUILD_VARIANT"] = variant.name
    
    return env

def build_cargo_bin(variant, release, package, output_name=None, targets=None):
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
            **rust_env(release=release, variant=variant),
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

def build_npm_packages():
    run_cmd(["pnpm", "install", "--frozen-lockfile"])
    run_cmd(["pnpm", "build"])
    
    # Copy to output
    dashboard_path = BUILD_DIR / "dashboard"
    shutil.rmtree(dashboard_path, ignore_errors=True)
    shutil.copytree("packages/dashboard-app/dist", dashboard_path)
    
    autocomplete_path = BUILD_DIR / "autocomplete"
    shutil.rmtree(autocomplete_path, ignore_errors=True)
    shutil.copytree("packages/autocomplete-app/dist", autocomplete_path)
    
    return dashboard_path, autocomplete_path

def build_linux_minimal(cli_path, pty_path):
    """Creates a minimal Linux archive with just the CLI and PTY binaries"""
    archive_name = LINUX_ARCHIVE_NAME
    archive_path = pathlib.Path(archive_name)
    archive_path.mkdir(parents=True, exist_ok=True)
    
    # Copy install script and README
    shutil.copy2("bundle/linux/install.sh", archive_path)
    shutil.copy2("bundle/linux/README", archive_path)
    
    # Create bin directory and copy binaries
    archive_bin_path = archive_path / "bin"
    archive_bin_path.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cli_path, archive_bin_path / CLI_BINARY_NAME)
    shutil.copy2(pty_path, archive_bin_path / PTY_BINARY_NAME)
    
    # Create archives
    tar_gz_path = BUILD_DIR / f"{archive_name}.tar.gz"
    run_cmd(["tar", "-czf", tar_gz_path, archive_path])
    print(f"Created archive at {tar_gz_path}")
    
    # Clean up
    shutil.rmtree(archive_path)
    
    return tar_gz_path

def build_macos_dmg(cli_path, pty_path, dashboard_path, autocomplete_path):
    """Creates a minimal macOS DMG with the CLI and PTY binaries"""
    # Create app structure
    app_path = BUILD_DIR / f"{APP_NAME}.app"
    shutil.rmtree(app_path, ignore_errors=True)
    
    # Create basic app structure
    (app_path / "Contents/MacOS").mkdir(parents=True, exist_ok=True)
    (app_path / "Contents/Resources").mkdir(parents=True, exist_ok=True)
    
    # Copy binaries
    shutil.copy2(cli_path, app_path / f"Contents/MacOS/{CLI_BINARY_NAME}")
    shutil.copy2(pty_path, app_path / f"Contents/MacOS/{PTY_BINARY_NAME}")
    
    # Copy resources
    shutil.copytree(dashboard_path, app_path / "Contents/Resources/dashboard")
    shutil.copytree(autocomplete_path, app_path / "Contents/Resources/autocomplete")
    
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
    dmg_path = BUILD_DIR / f"{DMG_NAME}.dmg"
    dmg_path.unlink(missing_ok=True)
    
    try:
        import dmgbuild
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
        print("dmgbuild not installed. Please install with: pip install dmgbuild")
        print("Alternatively, you can use the following command to create a DMG:")
        print(f"hdiutil create -volname '{APP_NAME}' -srcfolder {app_path} -ov -format UDZO {dmg_path}")
    
    return dmg_path

def main():
    parser = argparse.ArgumentParser(description="Build minimal Amazon Q installers")
    parser.add_argument("--platform", choices=["mac", "linux", "both"], default="both", 
                        help="Platform to build for")
    parser.add_argument("--variant", choices=["minimal", "full"], default="minimal",
                        help="Build variant")
    parser.add_argument("--release", action="store_true", default=True,
                        help="Build in release mode")
    
    args = parser.parse_args()
    variant = Variant.MINIMAL if args.variant == "minimal" else Variant.FULL
    
    # Create build directory
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    
    # Build binaries
    print(f"Building {CLI_PACKAGE_NAME}")
    cli_path = build_cargo_bin(
        variant=variant,
        release=args.release,
        package=CLI_PACKAGE_NAME,
        output_name=CLI_BINARY_NAME,
    )
    
    print(f"Building {PTY_PACKAGE_NAME}")
    pty_path = build_cargo_bin(
        variant=variant,
        release=args.release,
        package=PTY_PACKAGE_NAME,
        output_name=PTY_BINARY_NAME,
    )
    
    # Build platform-specific installers
    if args.platform in ["mac", "both"] and is_darwin():
        print("Building npm packages")
        dashboard_path, autocomplete_path = build_npm_packages()
        
        print("Building macOS DMG")
        dmg_path = build_macos_dmg(cli_path, pty_path, dashboard_path, autocomplete_path)
        print(f"macOS DMG created at: {dmg_path}")
    
    if args.platform in ["linux", "both"] and is_linux():
        print("Building Linux archive")
        tar_path = build_linux_minimal(cli_path, pty_path)
        print(f"Linux archive created at: {tar_path}")
    
    print("Build completed successfully!")

if __name__ == "__main__":
    main()
