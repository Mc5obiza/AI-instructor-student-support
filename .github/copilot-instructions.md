<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->
- [x] Verify that the copilot-instructions.md file in the .github directory is created. (Initialized checklist file.)

- [x] Clarify Project Requirements (User requested FastAPI backend, Streamlit frontend, data/model dirs.)
	<!-- Ask for project type, language, and frameworks if not specified. Skip if already provided. -->

- [x] Scaffold the Project (Created FastAPI/Streamlit scaffold with required folders.)
	<!--
	Ensure that the previous step has been marked as completed.
	Call project setup tool with projectType parameter.
	Run scaffolding command to create project files and folders.
	Use '.' as the working directory.
	If no appropriate projectType is available, search documentation using available tools.
	Otherwise, create the project structure manually using available file creation tools.
	-->

- [x] Customize the Project (Stubbed RAG processors, retriever, generator, analytics, and UI.)
	<!--
	Verify that all previous steps have been completed successfully and you have marked the step as completed.
	Develop a plan to modify codebase according to user requirements.
	Apply modifications using appropriate tools and user-provided references.
	Skip this step for "Hello World" projects.
	-->

- [x] Install Required Extensions (No extensions requested.)
	<!-- ONLY install extensions provided mentioned in the get_project_setup_info. Skip this step otherwise and mark as completed. -->

- [x] Compile the Project (Skipped: no build/compile step yet.)
	<!--
	Verify that all previous steps have been completed.
	Install any missing dependencies.
	Run diagnostics and resolve any issues.
	Check for markdown files in project folder for relevant instructions on how to do this.
	-->

- [x] Create and Run Task (Not needed for this scaffold.)
	<!--
	Verify that all previous steps have been completed.
	Check https://code.visualstudio.com/docs/debugtest/tasks to determine if the project needs a task. If so, use the create_and_run_task to create and launch a task based on package.json, README.md, and project structure.
	Skip this step otherwise.
	 -->

- [x] Launch the Project (User will run FastAPI/Streamlit manually.)
	<!--
	Verify that all previous steps have been completed.
	Prompt user for debug mode, launch only if confirmed.
	 -->

- [x] Ensure Documentation is Complete (README added; removed instructional comments.)
- Work through each checklist item systematically.
- Keep communication concise and focused.
- Follow development best practices.
