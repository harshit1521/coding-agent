# Minimal Coding Agent

A minimal command-line coding agent built with Go and OpenRouter. It uses
`nvidia/nemotron-3-ultra-550b-a55b:free` and currently has three tool calls:
- **read_file** - Read the contents of a given relative file path
- **list_files** - List files and directories at a given path (defaults to current directory)
- **edit_file** - Make edits to a text file by replacing `old_str` with `new_str` (creates file if it doesn't exist)

## Requirements

- Go 1.27.3 or later
- An OpenRouter API key

## Run

From the project directory, tidy the Go dependencies:

```sh
go mod tidy
```

Set your OpenRouter API key in the current terminal session, then start the
agent:

**Windows PowerShell**

```powershell
$env:OPENROUTER_API_KEY = "your-openrouter-api-key"
go run .
```

**macOS / Linux**

```sh
export OPENROUTER_API_KEY="your-openrouter-api-key"
go run .
```

Enter a prompt when asked. Use `Ctrl+C` to quit. Keep your API key private and
do not commit it to the repository.