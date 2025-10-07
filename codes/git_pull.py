import git
repo_path = ("https://github.com/bavi-tesh/transfer_learning")
local_dest = (r'C:\Users\mbavi\Downloads\vscode_ext\cloned1')
repo = git.Repo.clone_from(repo_path, local_dest)
