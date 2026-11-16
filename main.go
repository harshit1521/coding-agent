package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path"
	"path/filepath"
	"strings"

	openrouter "github.com/OpenRouterTeam/go-sdk"
	"github.com/OpenRouterTeam/go-sdk/models/components"
	"github.com/OpenRouterTeam/go-sdk/models/operations"
	"github.com/OpenRouterTeam/go-sdk/optionalnullable"
	"github.com/invopop/jsonschema"
)

func main() {
	client := openrouter.New(
		openrouter.WithSecurity(os.Getenv("OPENROUTER_API_KEY")),
	)

	scanner := bufio.NewScanner(os.Stdin)

	getUserMessage := func() (string, bool) {
		if !scanner.Scan() {
			return "", false
		}
		return scanner.Text(), true
	}

	tools := []ToolDefinition{ReadFileDefinition, ListFilesDefinition, EditFileDefinition}

	agent := NewAgent(client, getUserMessage, tools)

	if err := agent.Run(context.Background()); err != nil { 
		fmt.Printf("Error: %s\n", err)
	}
}
type Agent struct {
	client         *openrouter.OpenRouter
	getUserMessage func() (string, bool)
	tools          []ToolDefinition
}

func NewAgent(
	client *openrouter.OpenRouter,
	getUserMessage func() (string, bool),
	tools []ToolDefinition,
) *Agent {
	return &Agent{
		client:         client,
		getUserMessage: getUserMessage,
		tools:          tools,
	}
}

func (a *Agent) Run(ctx context.Context) error {
	conversation := []components.ChatMessages{}

	fmt.Println("Chat with OpenRouter (use 'ctrl-c' to quit)")
	readUserInput := true

	for {
		if readUserInput {
			fmt.Print("\033[94mYou\033[0m: ")

			userInput, ok := a.getUserMessage()
			if !ok {
				break
			}

			userMessage := components.CreateChatMessagesUser(
				components.ChatUserMessage{
					Role:    components.ChatUserMessageRoleUser,
					Content: components.CreateChatUserMessageContentStr(userInput),
				},
			)

			conversation = append(conversation, userMessage)
		}

		response, err := a.runInference(ctx, conversation)
		if err != nil {
			return err
		}

		if response == nil ||
			response.ChatResult == nil ||
			len(response.ChatResult.Choices) == 0 {
			return fmt.Errorf("OpenRouter returned an empty response")
		}

		assistantMessage := response.ChatResult.Choices[0].Message

		conversation = append(
			conversation,
			components.CreateChatMessagesAssistant(assistantMessage),
		)

		content, ok := assistantMessage.GetContent().Get()
		if ok && content != nil {
			switch content.Type {
			case components.ChatAssistantMessageContentTypeStr: // text content
				if content.Str != nil {
					fmt.Printf(
						"\033[93mAssistant\033[0m: %s\n",
						*content.Str,
					)
				}

			case components.ChatAssistantMessageContentTypeArrayOfChatContentItems: // multimodal content (e.g., text + images)
				fmt.Println("\033[93mAssistant\033[0m: [Multimodal content]")
			}
		}

		toolCalls := assistantMessage.GetToolCalls()
		toolResults := []components.ChatMessages{}

		for _, toolCall := range toolCalls {
			fn := toolCall.GetFunction()

			input := json.RawMessage(fn.GetArguments())

			if !json.Valid(input) {
				return fmt.Errorf(
					"invalid arguments returned for tool %s: %s",
					fn.GetName(),
					fn.GetArguments(),
				)
			}

			result := a.executeTool(
				toolCall.GetID(),
				fn.GetName(),
				input,
			)

			toolResults = append(toolResults, result)
		}

		if len(toolResults) == 0 {
			readUserInput = true
			continue
		}

		readUserInput = false
		conversation = append(conversation, toolResults...)
	}

	return nil
}

func (a *Agent) runInference(
	ctx context.Context,
	conversation []components.ChatMessages,
) (*operations.SendChatCompletionRequestResponse, error) {

	openRouterTools := []components.ChatFunctionTool{}

	for _, tool := range a.tools {
		var schema map[string]interface{}

		if err := json.Unmarshal(tool.InputSchema, &schema); err != nil {
			return nil, fmt.Errorf(
				"invalid schema for tool %s: %w",
				tool.Name,
				err,
			)
		}

		function := components.ChatFunctionToolFunctionFunction{
			Name:        tool.Name,
			Description: &tool.Description,
			Parameters:  schema,
		}

		openRouterTools = append(
			openRouterTools,
			components.CreateChatFunctionToolChatFunctionToolFunction(
				components.ChatFunctionToolFunction{
					Type:     components.ChatFunctionToolTypeFunction,
					Function: function,
				},
			),
		)
	}

	response, err := a.client.Chat.Send(
		ctx,
		components.ChatRequest{
			Model: openrouter.Pointer("nvidia/nemotron-3-ultra-550b-a55b:free"),
			MaxTokens: optionalnullable.From(
				openrouter.Pointer(int64(4096)),
			),
			Messages: conversation,
			Tools:    openRouterTools,
		},
		nil,
	)

	if err != nil {
		return nil, err
	}

	return response, nil
}

type ToolDefinition struct {
	Name        string
	Description string
	InputSchema json.RawMessage
	Function    func(input json.RawMessage) (string, error)
}

type OpenRouterTool struct {
	Type     string `json:"type"`
	Function struct {
		Name        string          `json:"name"`
		Description string          `json:"description"`
		Parameters  json.RawMessage `json:"parameters"`
	} `json:"function"`
}

// tools 

// 1. read_file
var ReadFileDefinition = ToolDefinition{
	Name:        "read_file",
	Description: "Read the contents of a given relative file path. Use this when you want to see what's inside a file. Do not use this with directory names.",
	InputSchema: ReadFileInputSchema,
	Function:    ReadFile,
}

type ReadFileInput struct {
	Path string `json:"path" jsonschema_description:"The relative path of a file in the working directory."`
}

var ReadFileInputSchema = GenerateSchema[ReadFileInput]()

func ReadFile(input json.RawMessage) (string, error) {
	readFileInput := ReadFileInput{}

	err := json.Unmarshal(input, &readFileInput)
	if err != nil {
		return "", err
	}

	content, err := os.ReadFile(readFileInput.Path)
	if err != nil {
		return "", err
	}

	return string(content), nil
}

// 2. list_file

var ListFilesDefinition = ToolDefinition{
	Name:        "list_files",
	Description: "List files and directories at a given path. If no path is provided, lists files in the current directory.",
	InputSchema: ListFilesInputSchema,
	Function:    ListFiles,
}

type ListFilesInput struct {
	Path string `json:"path,omitempty" jsonschema_description:"Optional relative path to list files from. Defaults to current directory if not provided."`
}

var ListFilesInputSchema = GenerateSchema[ListFilesInput]()

func ListFiles(input json.RawMessage) (string, error) {
	listFilesInput := ListFilesInput{}
	err := json.Unmarshal(input, &listFilesInput)
	if err != nil {
		panic(err)
	}

	dir := "."
	if listFilesInput.Path != "" {
		dir = listFilesInput.Path
	}

	var files []string
	
	err = filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		relPath, err := filepath.Rel(dir, path)
		if err != nil {
			return err
		}

		if relPath != "." {
			if info.IsDir() {
				files = append(files, relPath+"/")
			} else {
				files = append(files, relPath)
			}
		}
		return nil
	})

	if err != nil {
		return "", err
	}

	result, err := json.Marshal(files)
	if err != nil {
		return "", err
	}

	return string(result), nil
}

// 3. edit_files

var EditFileDefinition = ToolDefinition{
	Name: "edit_file",
	Description: `Make edits to a text file.

Replaces 'old_str' with 'new_str' in the given file. 'old_str' and 'new_str' MUST be different from each other.

If the file specified with path doesn't exist, it will be created.
`,
	InputSchema: EditFileInputSchema,
	Function:    EditFile,
}

type EditFileInput struct {
	Path   string `json:"path" jsonschema_description:"The path to the file"`
	OldStr string `json:"old_str" jsonschema_description:"Text to search for - must match exactly and must only have one match exactly"`
	NewStr string `json:"new_str" jsonschema_description:"Text to replace old_str with"`
}

var EditFileInputSchema = GenerateSchema[EditFileInput]()

func EditFile(input json.RawMessage) (string, error) {
	editFileInput := EditFileInput{}
	err := json.Unmarshal(input, &editFileInput)
	if err != nil {
		return "", err
	}

	if editFileInput.Path == "" || editFileInput.OldStr == editFileInput.NewStr {
		return "", fmt.Errorf("invalid input parameters")
	}

	content, err := os.ReadFile(editFileInput.Path)
	if err != nil {
		if os.IsNotExist(err) && editFileInput.OldStr == "" {
			return createNewFile(editFileInput.Path, editFileInput.NewStr)
		}
		return "", err
	}

	oldContent := string(content)
	newContent := strings.Replace(oldContent, editFileInput.OldStr, editFileInput.NewStr, -1)

	if oldContent == newContent && editFileInput.OldStr != "" {
		return "", fmt.Errorf("old_str not found in file")
	}

	err = os.WriteFile(editFileInput.Path, []byte(newContent), 0644)
	if err != nil {
		return "", err
	}

	return "OK", nil
}

func createNewFile(filePath, content string) (string, error) {
	dir := path.Dir(filePath)
	if dir != "." {
		err := os.MkdirAll(dir, 0755)
		if err != nil {
			return "", fmt.Errorf("failed to create directory: %w", err)
		}
	}

	err := os.WriteFile(filePath, []byte(content), 0644)
	if err != nil {
		return "", fmt.Errorf("failed to create file: %w", err)
	}

	return fmt.Sprintf("Successfully created file %s", filePath), nil
}

func GenerateSchema[T any]() json.RawMessage {
	reflector := jsonschema.Reflector{
		AllowAdditionalProperties: false,
		DoNotReference:            true,
	}

	var v T
	schema := reflector.Reflect(v)

	data, err := json.Marshal(schema) // go object -> json schema
	if err != nil {
		panic(fmt.Errorf("failed to generate schema: %w", err))
	}

	return data
}

func (a *Agent) executeTool(
	id, name string,
	input json.RawMessage,
) components.ChatMessages {
	var toolDef ToolDefinition
	var found bool

	for _, tool := range a.tools {
		if tool.Name == name {
			toolDef = tool
			found = true
			break
		}
	}

	var result string

	if !found {
		result = "tool not found"
	} else if toolDef.Function == nil {
		result = "tool has no implementation"
	} else {
		fmt.Printf("\033[92mtool\033[0m: %s(%s)\n", name, input)

		response, err := toolDef.Function(input)
		if err != nil {
			result = err.Error()
		} else {
			result = response
		}
	}

	return components.CreateChatMessagesTool(
		components.ChatToolMessage{
			Role:       components.ChatToolMessageRoleTool,
			ToolCallID: id,
			Content:    components.CreateChatToolMessageContentStr(result),
		},
	)
}
