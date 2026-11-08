"""The project ID (DESIGN.md §12.2, TESTBED.md item 11), on git repositories made in `tmp_path`."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from mms import project


def git(where: Path, *args: str) -> str:
    """Git for building a fixture: no user or system config, a fixed author."""
    env = dict(os.environ, GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1", GIT_AUTHOR_NAME="t",
               GIT_AUTHOR_EMAIL="t@example.com", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@example.com")
    done = subprocess.run(["git", "-C", str(where), *args], env=env, capture_output=True, text=True, encoding="utf-8")
    assert done.returncode == 0, f"git {' '.join(args)}: {done.stderr}"
    return done.stdout.strip()


def make_repo(path: Path, commits: int = 2) -> Path:
    path.mkdir(parents=True)
    git(path, "init", "-q", "-b", "main")
    for number in range(commits):
        (path / "file.txt").write_text(f"{path.name} {number}\n")
        git(path, "add", ".")
        git(path, "commit", "-q", "-m", f"commit {number}")
    return path


def root_commits(repo: Path) -> list[str]:
    return git(repo, "rev-list", "--max-parents=0", "HEAD").split()


def test_one_id_from_the_top_level_a_subfolder_a_worktree_a_clone_a_move_and_a_new_remote(tmp_path):
    repo = make_repo(tmp_path / "work" / "thing")
    (repo / "src" / "deep").mkdir(parents=True)
    expected = f"git:{root_commits(repo)[0][:16]}"

    top = project.find(repo)
    assert top.id == expected and top.name == "thing" and Path(top.root) == repo.resolve()
    sub = project.find(repo / "src" / "deep")
    assert (sub.id, sub.name, Path(sub.root)) == (expected, "thing", repo.resolve())

    git(repo, "worktree", "add", "-q", "--detach", str(tmp_path / "elsewhere"))
    assert project.find(tmp_path / "elsewhere").id == expected

    git(tmp_path, "clone", "-q", str(repo), str(tmp_path / "clone"))
    assert project.find(tmp_path / "clone").id == expected

    moved = tmp_path / "moved" / "renamed"
    moved.parent.mkdir()
    shutil.copytree(repo, moved)
    assert project.find(moved).id == expected
    assert project.find(moved).name == "renamed"  # the name follows the folder, the ID does not

    git(repo, "remote", "add", "origin", "https://github.com/a/b.git")
    assert project.find(repo).id == expected
    git(repo, "remote", "set-url", "origin", "git@example.org:other/place.git")
    assert project.find(repo).id == expected


def test_a_repository_with_several_root_commits_takes_the_smallest(tmp_path):
    repo = make_repo(tmp_path / "multi", commits=1)
    git(repo, "checkout", "-q", "--orphan", "other")
    (repo / "other.txt").write_text("other\n")
    git(repo, "add", "other.txt")
    git(repo, "commit", "-q", "-m", "second root")
    git(repo, "checkout", "-q", "main")
    git(repo, "merge", "-q", "--allow-unrelated-histories", "-m", "join", "other")
    roots = root_commits(repo)
    assert len(roots) == 2
    assert project.find(repo).id == f"git:{min(roots)[:16]}"


def test_a_shallow_clone_takes_the_remote_id_and_urls_of_other_forms_match(tmp_path):
    source = make_repo(tmp_path / "source", commits=3)
    ids = []
    for name, url in (("ssh", "git@github.com:Some/Repo.git"), ("https", "https://github.com/some/repo"),
                      ("full", "ssh://git@GitHub.com/some/repo.git/"), ("other", "https://github.com/some/else")):
        git(tmp_path, "clone", "-q", "--depth", "1", source.as_uri(), str(tmp_path / name))
        assert git(tmp_path / name, "rev-parse", "--is-shallow-repository") == "true"
        git(tmp_path / name, "remote", "set-url", "origin", url)
        found = project.find(tmp_path / name)
        assert found.id.startswith("remote:") and len(found.id) == len("remote:") + 16
        ids.append(found.id)
    assert ids[0] == ids[1] == ids[2] != ids[3]
    assert ids[0] != project.find(source).id  # the full repository is not shallow


def test_a_shallow_clone_without_an_origin_has_no_project(tmp_path):
    source = make_repo(tmp_path / "source", commits=3)
    git(tmp_path, "clone", "-q", "--depth", "1", source.as_uri(), str(tmp_path / "shallow"))
    git(tmp_path / "shallow", "remote", "remove", "origin")
    assert project.find(tmp_path / "shallow") is None


@pytest.mark.parametrize("url, normal", [
    ("git@github.com:a/b.git", "github.com/a/b"),
    ("https://github.com/a/b", "github.com/a/b"),
    ("https://user:secret@GitHub.com/A/B.git/", "github.com/a/b"),
    ("ssh://git@host.example:2222/a/b.git", "host.example:2222/a/b"),
    ("git://github.com/a/b.git", "github.com/a/b"),
])
def test_remote_urls_are_normalized(url, normal):
    assert project.normalize_url(url) == normal


def test_a_marker_file_wins_over_git_from_its_folder_and_from_below(tmp_path):
    repo = make_repo(tmp_path / "repo")
    (repo / "a" / "b").mkdir(parents=True)
    (repo / "a" / ".mms-project").write_text("shared-id\n")
    assert project.find(repo).id.startswith("git:")  # above the marker: git's
    assert project.find(repo / "a").id == "id:shared-id"
    assert project.find(repo / "a" / "b").id == "id:shared-id"
    (repo / ".mms-project").write_text("topid\n")  # at the top level: for the whole repository
    assert project.find(repo).id == "id:topid"
    assert project.find(repo / "a").id == "id:shared-id"  # the nearest wins


def test_a_marker_above_the_git_top_level_is_not_read(tmp_path):
    (tmp_path / "outer").mkdir()
    (tmp_path / "outer" / ".mms-project").write_text("outer-id")
    repo = make_repo(tmp_path / "outer" / "repo")
    assert project.find(repo).id.startswith("git:")


def test_a_marker_works_outside_git_in_its_own_folder_only(tmp_path):
    plain = tmp_path / "plain"
    (plain / "sub").mkdir(parents=True)
    (plain / ".mms-project").write_text("plain-id")
    found = project.find(plain)
    assert (found.id, found.name) == ("id:plain-id", "plain")
    assert project.find(plain / "sub") is None


def test_a_bad_marker_is_an_error_not_a_project(tmp_path):
    (tmp_path / ".mms-project").write_text("no way/ not an id")
    with pytest.raises(ValueError, match="mms-project"):
        project.find(tmp_path)


def test_a_folder_outside_git_and_a_repository_without_commits_have_no_project(tmp_path):
    (tmp_path / "plain").mkdir()
    assert project.find(tmp_path / "plain") is None
    empty = tmp_path / "empty"
    empty.mkdir()
    git(empty, "init", "-q")
    assert project.find(empty) is None


def test_the_folder_is_claude_project_dir_else_the_working_directory(tmp_path, monkeypatch):
    monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
    assert project.folder() == Path.cwd()
    monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(tmp_path))
    assert project.folder() == tmp_path
