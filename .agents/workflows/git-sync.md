---
description: sync NAS project files to GitHub via local git clone
---

# Git Sync Workflow

NAS 不支持 git 写操作，所有 git 操作都通过本地 clone `/tmp/doa_repo` 完成。

## Variables
- NAS_DIR: `/Volumes/personal_folder/NAS_Documents/MD9120/DOA-CNN-TCA-ResNeXt`
- LOCAL_CLONE: `/tmp/doa_repo`
- REMOTE: `https://github.com/eastshg365-cmd/DOA-CNN-TCA-ResNeXt.git`
- GH_TOKEN: 由用户提供 (格式: `ghp_xxxx`)

---

## Push (NAS → GitHub)

// turbo
1. 确保本地 clone 存在，如果不存在则重新克隆：
```bash
[ -d /tmp/doa_repo/.git ] || git clone https://github.com/eastshg365-cmd/DOA-CNN-TCA-ResNeXt.git /tmp/doa_repo
```

// turbo
2. 从 NAS 同步文件到本地 clone（排除大文件和生成文件）：
```bash
rsync -av --delete \
  --exclude='.git/' \
  --exclude='*.cdf' \
  --exclude='HTMLFiles/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='.DS_Store' \
  /Volumes/personal_folder/NAS_Documents/MD9120/DOA-CNN-TCA-ResNeXt/ \
  /tmp/doa_repo/
```

// turbo
3. 检查有什么变化：
```bash
git -C /tmp/doa_repo status --short
```

4. 如果有变化，commit 并 push（需要用户提供 token）：
```bash
cd /tmp/doa_repo
git add -A
git commit -m "chore: sync from NAS $(date '+%Y-%m-%d %H:%M')"
git remote set-url origin "https://<GH_TOKEN>@github.com/eastshg365-cmd/DOA-CNN-TCA-ResNeXt.git"
git push origin main
git remote set-url origin "https://github.com/eastshg365-cmd/DOA-CNN-TCA-ResNeXt.git"
```

---

## Pull (GitHub → NAS)

// turbo
1. 确保本地 clone 存在：
```bash
[ -d /tmp/doa_repo/.git ] || git clone https://github.com/eastshg365-cmd/DOA-CNN-TCA-ResNeXt.git /tmp/doa_repo
```

// turbo
2. 拉取最新：
```bash
git -C /tmp/doa_repo remote set-url origin "https://<GH_TOKEN>@github.com/eastshg365-cmd/DOA-CNN-TCA-ResNeXt.git"
git -C /tmp/doa_repo pull origin main
git -C /tmp/doa_repo remote set-url origin "https://github.com/eastshg365-cmd/DOA-CNN-TCA-ResNeXt.git"
```

// turbo
3. 将最新内容同步回 NAS（只同步代码文件，不覆盖 NAS 独有的笔记本等）：
```bash
rsync -av \
  --exclude='.git/' \
  --exclude='img/' \
  --exclude='*.nb' \
  --exclude='*.wl' \
  --exclude='*.pdf' \
  /tmp/doa_repo/ \
  /Volumes/personal_folder/NAS_Documents/MD9120/DOA-CNN-TCA-ResNeXt/
```
