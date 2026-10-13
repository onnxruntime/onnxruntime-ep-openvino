# Contributing to Intel® OpenVINO™ Execution Provider (EP) Plugin

We welcome contributions to the **Intel® OpenVINO™ Execution Provider (EP) plugin** project from both Intel employees and the open-source community.
This document provides guidelines to help you make successful contributions and ensure a smooth review and integration process.

## Table of Contents
1. [Ways to Contribute](#ways-to-contribute)
2. [Code Contribution Process](#code-contribution-process)
3. [Pull Request Requirements](#pull-request-requirements)
4. [Review Process](#review-process)
5. [Responsibilities](#responsibilities)
6. [Additional Resources](#additional-resources)


## Ways to Contribute

See also: [AI Usage Policy](./AI_USAGE_POLICY.md).
If you use AI assistance for any contribution activity (code, docs, issues,
discussions, or PR communication), read and follow this policy.

### Report Bugs or Propose Features

- If you experience unexpected behavior or have ideas for improvement:
  - Submit a [GitHub issue](https://github.com/onnxruntime/onnxruntime-ep-openvino/issues) with a clear description, logs, and steps to reproduce.
  - Or join the [GitHub Discussions](https://github.com/onnxruntime/onnxruntime-ep-openvino/discussions) if it's exploratory or architectural.

### Suggest Architecture Improvements

- For major proposals or POCs, open a **Draft Pull Request** and mark it as such to initiate discussion without triggering full review flow.

### Contribute Code

- Start from open issues or tasks in the [issue tracker](https://github.com/onnxruntime/onnxruntime-ep-openvino/issues).
- Follow our [Pull Request Process](#code-contribution-process).


## Code Contribution Process

The Intel® OpenVINO™ Execution Provider (EP) plugin project contribution process is detailed in [CONTRIBUTING_PR.md](./CONTRIBUTING_PR.md). We recommend familiarizing yourself with it before submitting a PR.

1. **Fork the Repository**
   All contributions must come from **forks**, not branches in the main repository.

2. **Create a Draft PR for Early Feedback**
   Use draft pull requests to share early ideas, work-in-progress, or architectural proposals.

3. **Mark as Ready for Review**
   Remove "Draft" status when your contribution is ready and meets the [PR Requirements](#pull-request-requirements).

4. **Review and Verification**
   Address reviewer feedback, rebase your branch, and verify your changes.

5. **Merge and Integration**
   Once approved and validated, maintainers will proceed with internal integration and merging.


## Pull Request Requirements

Your PR must:

✅ **Target a single purpose (Feature / Bug / Refactor / etc.)**
✅ Include a meaningful title, summary, and description (reference issues with parentheses, e.g. `Summary (#123)`)
✅ Be rebased on the target branch (no merge commits)
✅ Contain logically separated commits with descriptions (avoid "squash-all" style)
✅ Add test coverage for all new or affected behavior
✅ Contain **validation results** (local test run results, build logs, or test command outputs)

## Review Process

- Request initial review from maintainers and domain experts
- Make sure **all discussions are resolved**, especially blocking ones
- PR will be merged once:
  - Reviewers have approved
  - Test validation results are verified
  - Discussions are closed
- Maintainer performs final review and merge

> **Note on Upstream Integration:** This repository contains release snapshots. Accepted pull requests may be integrated and validated through internal development pipelines before being merged into the `main` branch of this repository.

> **Note:** For large PRs (e.g. refactor, submodule bump), mark it explicitly in the header so maintainers can manage merge conflicts accordingly.

## Responsibilities

### Pull Request Author
- Own the PR lifecycle from creation to merge
- Rebase, resolve comments, and ensure validation
- Escalate blockers proactively
- Ensure functionality lands in the correct branches

### Reviewer
- Provide timely, constructive feedback
- Approve only when confident in change scope and correctness

### Maintainer
- Confirm all merge conditions are satisfied
- Avoid conflicts with other recent changes
- Merge only when verification and review criteria are met

## Additional Resources

- [Code of Conduct](./CODE_OF_CONDUCT.md)
- [Pull Request Guidelines](./CONTRIBUTING_PR.md)
- [AI Usage Policy](./AI_USAGE_POLICY.md)
- [How to fork a repository](https://docs.github.com/en/get-started/quickstart/fork-a-repo)

💬 **Got Questions?**
Submit a [GitHub issue](https://github.com/onnxruntime/onnxruntime-ep-openvino/issues), start a [GitHub Discussion](https://github.com/onnxruntime/onnxruntime-ep-openvino/discussions), or reach out to the development team through internal channels (for Intel employees).


---

By contributing to this repository, you agree that your work will be licensed under the terms of the [LICENSE](./LICENSE).
