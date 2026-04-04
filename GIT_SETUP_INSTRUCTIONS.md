# Git Setup Instructions for AITrading

Git is not available in the current environment. To push this project to GitHub, run these commands in PowerShell or Command Prompt **on a machine with Git installed**:

## Quick Setup (Copy & Paste)

```powershell
cd C:\Workspace\AITrading
echo "# AITrading" >> README.md
git init
git add .
git config user.email "you@example.com"
git config user.name "Your Name"
git commit -m "Initial commit: AITrading backtester project"
git branch -M main
git remote add origin https://github.com/XnitishX/AITrading.git
git push -u origin main
```

## Step-by-Step Explanation

1. **Add README header** (if not already present)
   ```
   echo "# AITrading" >> README.md
   ```

2. **Initialize local git repository**
   ```
   git init
   ```

3. **Stage all files** (`.gitignore` is already in place to exclude venv, __pycache__, etc.)
   ```
   git add .
   ```

4. **Set git user config** (use your actual email/name)
   ```
   git config user.email "you@example.com"
   git config user.name "Your Name"
   ```

5. **Create initial commit**
   ```
   git commit -m "Initial commit: AITrading backtester project"
   ```

6. **Rename branch to main** (if on older git that defaults to master)
   ```
   git branch -M main
   ```

7. **Add remote origin**
   ```
   git remote add origin https://github.com/XnitishX/AITrading.git
   ```

8. **Push to GitHub**
   ```
   git push -u origin main
   ```

## Prerequisites

- **Git** must be installed. Download from: https://git-scm.com/download/win
- **GitHub account** with the repository `XnitishX/AITrading` created
- SSH key or personal access token (PAT) configured for authentication

## Notes

- A `.gitignore` file is already in place to exclude virtual environments, caches, databases, and logs
- The project structure is preserved with all source code, data pipeline, simulator, and web API
