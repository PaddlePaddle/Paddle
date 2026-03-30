---
name: paddle-pull-request
description: |
  自动创建或更新 GitHub Pull Request。
  当需要为 Paddle 相关仓库创建 PR 时，优先使用本 skill。
---

# Paddle 仓库 PR 创建与更新

## 流程

### 1. 检查分支状态

- 检查当前分支是否已经推送到远端；如果没有，执行 `git push -u origin HEAD`。
- 如果当前分支名是 `main` 或 `master`，在继续之前先向用户确认是否真的要在该分支上直接提 PR。

### 2. 按逻辑主题整理改动

- 不要机械地罗列每一次 commit。
- 按照「功能 / 目的」对改动进行分组，回答：
  - 为什么需要这次改动？
  - 解决了什么问题？
  - 大致改了哪些模块？

### 3. 使用 Paddle 官方 PR 模板

- PR 内容必须遵循 Paddle 官方 PR 模板：
  - 模板链接：`https://github.com/PaddlePaddle/Paddle/blob/develop/.github/PULL_REQUEST_TEMPLATE.md`
  - 模板使用**三级标题（`###`）**作为各小节标题，生成 PR 描述时需与之保持一致。
  - 模板结构（简化版）：

```markdown
### PR Category
<!-- One of [ User Experience | Execute Infrastructure | Operator Mechanism | CINN | Custom Device | Performance Optimization | Distributed Strategy | Parameter Server | Communication Library | Auto Parallel | Inference | Environment Adaptation ] -->

### PR Types
<!-- One of [ New features | Bug fixes | Improvements | Performance | BC Breaking | Deprecations | Docs | Devs | Not User Facing | Security | Others ] -->

### Description

### 是否引起精度变化
<!-- one of the following [ 是 | 否 ]-->
```

- 生成 PR 描述时，按以上四个小节依次填写：
  - **PR Category**：高层次类别，例如 Bug fix、Feature、Refactor、Doc 等。
  - **PR Types**：更细的类型说明，例如 API 变更、性能优化、算子新增等。
  - **Description**：用自然语言简要说明该 PR 的背景、动机和主要改动点。**描述正文中不要使用三级及三级以上标题**（`###`、`##`、`#`），若需分段小标题，最多使用四级标题（`####`）。
  - **是否引起精度变化**：明确说明是否会影响已有模型或任务的精度，并给出必要的说明。

### 4. 提交前运行 prek 检查

- 在创建或更新 PR 之前，先运行 `prek` 对所有文件进行格式检查：

  ```bash
  prek
  ```

- 如果 `prek` 自动修改了文件，需要将这些修改加入一个新的 commit，再推送：

  ```bash
  git add .
  git commit -m "🔧 chore: apply prek fixes"
  git push
  ```

- 只有 `prek` 检查通过（没有额外修改）后，再继续创建或更新 PR。
- 如果 `prek` 报告无法自动修复的错误（例如类型检查或自定义 hook 失败），需根据错误信息手动修复后重新运行，直到全部检查通过，再继续创建或更新 PR。

### 5. 使用 gh 命令创建 / 更新 PR

- 使用 `gh` 命令创建或更新 PR。

#### 标题规范

- 标题整体用英文，保持简洁明了。
- 推荐格式：`[PR 大类] 简要说明`
  - 示例：`[CINN] avoid wrong fusion for xxx op`
  - 示例：`[LargeTensor] fix xxx kernel`
  - 示例：`[CodeStyle] update code style`
- 避免使用含糊标题，例如：
  - `fix bug` / `update code` / `test` / `temp` / `WIP` 等。
- 尽量控制在一行内说清「做了什么」或「修改目的」，不需要罗列所有细节。

示例命令（根据实际情况替换标题和正文）：

```bash
gh pr create --title "[xxx] xxx" --body "$(cat <<'EOF'
### PR Category
Operator Mechanism

### PR Types
New features

### Description
在这里说明该改动的动机和主要变化，可根据实际情况扩展，优先使用中文。

### 是否引起精度变化
否

EOF
)"
```

## 注意事项

- 始终使用 Paddle 官方 PR 模板的章节结构，**标题层级与模板一致**（模板使用三级标题 `###`），不要自定义新的顶层标题。
- **Description 描述正文**中不要加三级及三级以上标题，若有分段需求，最多使用四级标题（`####`）。
- 优先强调「为什么需要这次改动」，而不是罗列所有实现细节。
- 如果业务或背景信息不清楚，应先向用户提问澄清，再生成 PR 描述。
- 成功创建或更新 PR 后，应返回 PR URL，方便用户查看。
 - 为 Paddle 相关仓库生成 PR 描述、调试说明或问题分析时，**优先使用中文**，除非上游模板或明确场景要求使用英文。
 - **不要将任何 bot / 机器账号加入协作者**。
