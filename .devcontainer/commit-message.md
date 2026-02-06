<!--
Source - https://stackoverflow.com/q
Posted by fingers10, modified by community. See post 'Timeline' for change history
Retrieved 2025-11-18, License - CC BY-SA 4.0
settings.json:

{
	"github.copilot.chat.commitMessageGeneration.instructions": [
		{
			"file": ".devcontainer/commit-message.md"
		}
	],
}

-->

Write a descriptive commit message based solely on the provided changes in chinese. Follow these strict rules:


Use the correct keyword from the provided list to represent the type of change.
Separate subject from body with a blank line.
Use the body to explain what and why you have done something. In most cases, you can leave out details about how a change has been made.
In the body, use bullet points to describe everything.
Avoid vague terms like "update", "enhance", "improve", or "better".
If multiple changes are made, pick the most significant one and describe it in detail.
Message length is maximum 250 characters.
keywords: 

WIP: Work in progress.
Performance: Improve performance.
Bug: Fix a bug.
Feature: Introduce new feature.
Docs: Add or update documentation.
Test: Add, update, or pass tests.
Security: Fix security issues.
Refactor: Refactor code.
