# How to Prepare a Good PR

   Intel® OpenVINO™ Execution Provider (EP) plugin is an open-source project and you can contribute to its code directly.
   To do so, follow these guidelines for creating Pull Requests, so that your
   changes get the highest chance of being merged.


## General Rules of a Good Pull Request

* Create your own fork of the repository and use it to create PRs.
  Avoid creating change branches in the main repository.
* Choose a proper branch for your work and create your own branch based on it.
* Give your branches, commits, and Pull Requests meaningful names and descriptions.
  It helps to track changes later.
  * If referencing a GitHub issue, suffix or prefix the issue number in parentheses in the commit summary and PR title (e.g., `Add feature X (#123)` or `(#123) Add feature X`), so it links automatically and shows up clearly in `git log`.
* Follow the project's coding conventions:
  * **C++ Standard**: C++23 for both implementation and tests.
  * **Code Formatting**: Google C++ style enforced by `.clang-format` (120-character column limit). Run `clang-format` before submitting.
  * **Naming**: `PascalCase` for types, `snake_case` for variables and functions, `UPPER_SNAKE_CASE` with `OVEP_` prefix for macros.
  * **Includes**: Use `#pragma once` for header guards.
  * **Namespaces**: All non-test code belongs to `onnxruntime::openvino_ep_plugin`.
  * **OpenVINO API**: Use OpenVINO 2.0+ API (`ov::` namespace). Legacy `InferenceEngine::` / `ie_` APIs are not allowed.
  * **Resource Management**: Prefer `std::unique_ptr` / `std::shared_ptr` over raw owning pointers; follow RAII and the Rule of Five.
  * **Error Handling**: Use `OVEP_API_ENTRY` for ORT C API callbacks, never allow exceptions to escape C boundary callbacks, and propagate statuses using `OVEP_RETURN_IF_ERROR`.
  * **Path Handling**: Use `PathFromUtf8` / `ToUtf8String` for cross-platform UTF-8 filesystem path conversions.
* Make your PRs small - each PR should address one issue. Remove all changes
  unrelated to the PR.
* [Link your Pull Request to an issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue#manually-linking-a-pull-request-to-an-issue-using-the-pull-request-sidebar) if it addresses one.
* Document your contribution! If your changes impact how users or developers interact with the plugin, provide the information in the appropriate documentation files.
* Every new file must include the required license and copyright header (MIT License / Intel Corporation).
* Follow the [AI Usage Policy](./AI_USAGE_POLICY.md) and disclose significant AI assistance in your PR description.
* For Work In Progress, or checking test results early, use a Draft PR.


## Ensure Change Quality

Pull requests will undergo review and validation. Ensure the quality of your changes before submitting:

* Test your changes locally:
  * Double-check your code and run `clang-format` before committing.
  * Run tests locally using CTest or the `ovep_tests` test binary from the build/install artifacts directory.
  * Include test results or validation output in your pull request description.
* Before creating a PR, make sure that your branch is up to date with the latest
  state of the target branch (e.g. `git fetch upstream && git rebase upstream/main`).


## Branching Policy

* Contributions should target the `main` branch for upcoming features and improvements.
* **Release Snapshots & Integration**: As this public repository reflects release snapshots, incoming pull requests may be merged and validated through internal development pipelines first and subsequently merged into the `main` branch of this repository on next release.


## Pull Request Activity and Review Process

### What is considered activity

A Pull Request (PR) is considered **active** when there is meaningful progress, such as:

* Code updates addressing review feedback
* Technical discussion resolving open questions
* Significant rework or design clarification

Simple reminders, “ping” messages, or trivial updates (e.g. rebasing without changes) are not considered meaningful activity.


### Response expectations

After review feedback is provided:

* Contributors are expected to respond within **21 days**
* If there is no response, the PR may be considered **inactive**

This helps ensure that review bandwidth is focused on contributions that are actively progressing.


### Stale Pull Requests

If a PR becomes inactive:

* A maintainer may mark it with a `stale` label
* A reminder comment may be added requesting an update

This is a signal that the PR requires attention to continue review.


### Closing inactive Pull Requests

If there is no meaningful activity within **7 days after being marked as stale**:

* The PR may be **closed by maintainers**

Closing a PR due to inactivity:

* Is an **administrative action**, not a rejection
* Does **not prevent resubmission** of the contribution later


### Continuing or reviving work

If a PR has been closed due to inactivity:

* Contributors are welcome to reopen it (if possible), or
* Submit a new PR referencing the previous work

Maintainers will resume the review once activity continues.


### Maintainer continuation

To avoid losing valuable contributions:

* If a PR is inactive but relevant, maintainers may:
  * Continue work directly on top of the PR, or
  * Create a follow-up PR based on the original contribution

In such cases, credit to the original author will be preserved.


### Community reviews

We encourage community participation in the review process.

* PRs are open for review by external contributors
* Constructive feedback from the community is considered during merge decisions

Community contributions are highly valued and help improve review scalability.


## Need Additional Help? Check these Articles

* [How to fork a repository](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo)
* [Install Git](https://git-scm.com/book/en/v2/Getting-Started-First-Time-Git-Setup)
* [Contributing Guide](./CONTRIBUTING.md)
* [AI Usage Policy](./AI_USAGE_POLICY.md)
* [Code of Conduct](./CODE_OF_CONDUCT.md)
