# Minimal Coding Agent

A minimal command-line coding agent built with Go and OpenRouter. It uses
`nvidia/nemotron-3-ultra-550b-a55b:free` and currently has one tool call:
`read_file`, which lets the model read a file in the working directory.

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