"""Generic code tools exposed to the open-ended Planner."""

from agents import RunContextWrapper, function_tool

from infra_mas.code_tasks.context import CodePlannerContext


@function_tool
async def search_code(
    ctx: RunContextWrapper[CodePlannerContext],
    query: str,
    path: str = ".",
) -> str:
    """Search tracked repository text for a literal query.

    Args:
        query: Literal symbol, error text, or code fragment to find.
        path: Repository-relative file or directory to restrict the search.
    """
    return await ctx.context.invoke_tool(
        "search_code", {"query": query, "path": path, "max_results": 50}
    )


@function_tool
async def read_file(
    ctx: RunContextWrapper[CodePlannerContext],
    path: str,
    start_line: int = 1,
    end_line: int = 200,
) -> str:
    """Read a bounded line range from one repository file.

    Args:
        path: Repository-relative text file path.
        start_line: First one-based line to return.
        end_line: Last one-based line to return, at most 500 lines after start_line.
    """
    return await ctx.context.invoke_tool(
        "read_file",
        {"path": path, "start_line": start_line, "end_line": end_line},
    )


@function_tool
async def edit_file(
    ctx: RunContextWrapper[CodePlannerContext],
    path: str,
    old_text: str,
    new_text: str,
) -> str:
    """Replace one exact, unique text block in a repository file.

    Args:
        path: Repository-relative UTF-8 file path.
        old_text: Exact existing text that must occur exactly once.
        new_text: Replacement text.
    """
    return await ctx.context.invoke_tool(
        "edit_file", {"path": path, "old_text": old_text, "new_text": new_text}
    )


@function_tool
async def apply_patch(
    ctx: RunContextWrapper[CodePlannerContext],
    patch: str,
) -> str:
    """Apply a unified Git diff to the repository after validation.

    Args:
        patch: Complete unified diff beginning with one or more diff --git headers.
    """
    return await ctx.context.invoke_tool("apply_patch", {"patch": patch})


@function_tool
async def run_targeted_test(
    ctx: RunContextWrapper[CodePlannerContext],
    test_path: str,
) -> str:
    """Run the configured test harness against one repository-relative test target.

    Args:
        test_path: Existing test file or directory relative to the repository root.
    """
    return await ctx.context.invoke_tool("run_targeted_test", {"test_path": test_path})


@function_tool
async def run_full_test(ctx: RunContextWrapper[CodePlannerContext]) -> str:
    """Run the Worker's fixed full repository test command."""
    return await ctx.context.invoke_tool("run_full_test", {})


@function_tool
async def submit_patch(ctx: RunContextWrapper[CodePlannerContext]) -> str:
    """Submit the current non-empty Git diff as the final benchmark patch."""
    return await ctx.context.invoke_tool("submit_patch", {})
