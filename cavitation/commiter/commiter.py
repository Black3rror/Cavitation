import subprocess
import time


def _run_git_command(*args):
    try:
        result = subprocess.run(['git', *args], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.stdout.decode().strip()
    except subprocess.CalledProcessError as e:
        print(f"Error executing `git {' '.join(args)}`:\n{e.stderr.decode().strip()}")
        exit(1)


def commit_experiment():
    """Tries to find a commit with the same repository state, if unsuccessful, creates a new branch
    and commits the current state of the repository there.

    Returns:
        tuple: A tuple containing the experiment commit message and its hash.
    """

    # Logic of the code:
    # 1. Current branch is clean, return its last commit message and hash
    # 2. Another branch with no difference exists, return that branch's last commit message and hash
    # 3. Creating a new branch is required:
    #   3.1. Stash all changes
    #   3.2. Create a new branch
    #   3.3. Apply the stash in the new branch and commit all changes
    #   3.4. Go back to the initial branch and pop the stash there

    # BUG: When adding a new file or renaming a file (which might be considered as deleting the old file and creating a new one),
    # the diff command will return a non-empty string, even if the file is the same. This will cause the code to create a new branch
    # and commit the changes there, even though the changes are not real

    time_tag = int(time.time()*1000)
    stash_name = "stash_{}".format(time_tag)
    new_branch = "checkpoint/branch_{}".format(time_tag)
    commit_message = "commit_{}".format(time_tag)

    if _run_git_command('status', '--porcelain') == '':
        # 1. Current branch is clean, return its last commit message and hash
        exp_commit_message = _run_git_command('log', '-1', '--pretty="%s"')
        exp_commit_hash = _run_git_command('log', '-1', '--pretty="%H"')
        return exp_commit_message, exp_commit_hash

    branches = _run_git_command('branch', '--contains', 'HEAD').split('\n')
    branches = [x for x in branches if '*' not in x]
    branches = [x.strip() for x in branches]
    for branch in branches:
        if _run_git_command('diff', branch) == '':
            # 2. Another branch with no difference exists, return that branch's last commit message and hash
            exp_commit_message = _run_git_command('log', '-1', '--pretty="%s"', branch)
            exp_commit_hash = _run_git_command('log', '-1', '--pretty="%H"', branch)
            return exp_commit_message, exp_commit_hash

    # 3.1. Stash all changes
    _run_git_command('stash', 'push', '--include-untracked', '-m', stash_name)

    # 3.2. Create a new branch
    _run_git_command('branch', new_branch)

    # 3.3. Apply the stash in the new branch and commit all changes
    initial_branch = _run_git_command('rev-parse', '--abbrev-ref', 'HEAD')
    _run_git_command('checkout', new_branch)

    _run_git_command('stash', 'apply', 'stash^{/'+stash_name+'}')
    _run_git_command('add', '.')
    _run_git_command('commit', '-m', commit_message, '--no-verify')
    exp_commit_message = _run_git_command('log', '-1', '--pretty="%s"')
    exp_commit_hash = _run_git_command('log', '-1', '--pretty="%H"')

    # 3.4. Go back to the initial branch and pop the stash there
    _run_git_command('checkout', initial_branch)
    _run_git_command('stash', 'pop')

    return exp_commit_message, exp_commit_hash

if __name__ == '__main__':
    commit_message, commit_hash = commit_experiment()
    print("Commit message: ", commit_message)
    print("Commit hash: ", commit_hash)
