# Development Scripts

This directory contains scripts for development and building minimal versions of Amazon Q CLI.

## Build Minimal Scripts

These scripts create minimal builds of Amazon Q CLI with only the essential components. The builds are optimized for both Intel and Apple Silicon on macOS.

### Prerequisites

#### For macOS:
- Python 3.6+
- Rust toolchain with both x86_64 and aarch64 targets:
  ```bash
  rustup target add x86_64-apple-darwin aarch64-apple-darwin
  ```
- For DMG creation (optional):
  ```bash
  pip install dmgbuild
  ```

#### For Linux:
- Python 3.6+
- Rust toolchain
- Build dependencies:
  ```bash
  sudo apt update
  sudo apt install build-essential pkg-config jq dpkg curl wget cmake
  ```

### Running the Scripts

#### For macOS:

The macOS-specific script builds a minimal version of Amazon Q CLI and packages it into a DMG:

```bash
# From the project root directory
python development-scripts/build_minimal_mac.py
```

This will:
1. Build the `q_cli` and `figterm` packages in release mode
2. Create universal binaries (for both Intel and Apple Silicon)
3. Package them into a minimal `.app` bundle
4. Create a DMG installer

The resulting DMG will be in the `build/` directory.

#### For Linux:

```bash
# From the project root directory
python development-scripts/build_minimal.py --platform linux
```

This will:
1. Build the `q_cli` and `figterm` packages in release mode
2. Package them into a tar.gz archive

The resulting archive will be in the `build/` directory.

#### For both platforms:

The main script supports building for both platforms:

```bash
# From the project root directory
python development-scripts/build_minimal.py --platform both
```

### Command Line Options

The main `build_minimal.py` script supports the following options:

```
usage: build_minimal.py [-h] [--platform {mac,linux,both}] [--variant {minimal,full}] [--release]

Build minimal Amazon Q installers

optional arguments:
  -h, --help            show this help message and exit
  --platform {mac,linux,both}
                        Platform to build for
  --variant {minimal,full}
                        Build variant
  --release             Build in release mode (default: True)
```

## Notes

- The minimal builds include only the CLI (`q`) and terminal (`qterm`) binaries
- The builds are optimized for performance with release mode and LTO enabled
- On macOS, universal binaries are created to support both Intel and Apple Silicon
- The macOS DMG includes a minimal app bundle structure
- The Linux build creates a simple tar.gz archive with the binaries
