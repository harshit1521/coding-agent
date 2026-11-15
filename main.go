package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os"

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

	tools := []ToolDefinition{ReadFileDefinition}

	agent := NewAgent(client, getUserMessage, tools)

	if err := agent.Run(context.Background()); err != nil {
		fmt.Printf("Error: %s\n", err)
	}
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

type Agent struct {
	client         *openrouter.OpenRouter
	getUserMessage func() (string, bool)
	tools          []ToolDefinition
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
			case components.ChatAssistantMessageContentTypeStr:
				if content.Str != nil {
					fmt.Printf(
						"\033[93mAssistant\033[0m: %s\n",
						*content.Str,
					)
				}

			case components.ChatAssistantMessageContentTypeArrayOfChatContentItems:
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

func GenerateSchema[T any]() json.RawMessage {
	reflector := jsonschema.Reflector{
		AllowAdditionalProperties: false,
		DoNotReference:            true,
	}

	var v T
	schema := reflector.Reflect(v)

	data, err := json.Marshal(schema)
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
