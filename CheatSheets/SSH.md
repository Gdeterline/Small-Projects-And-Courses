# Cheat Sheet: SSH (Secure Shell)

## Basic SSH Commands

### 1. **Connecting to a Remote Server:**

```bash
ssh username@hostname
```

#### Explanation:
- `ssh`: The command to initiate an SSH connection.
- `username`: Your username on the remote server.
- `hostname`: The IP address or domain name of the remote server.

#### Note:
- It may be necessary to specify the path to the private key in case of key-based authentication:
```bash
ssh -i /path/to/private_key username@hostname
```


### 2. **Copying Files with SCP:**

```bash
scp localfile username@hostname:/remote/directory
```

or

```bash
rsync -avz localfile username@hostname:/remote/directory
```

#### Explanation:
- `scp`: Secure copy command to transfer files between hosts.
- `localfile`: The file on your local machine that you want to copy.
- `username@hostname:/remote/directory`: The destination path on the remote server.

Rsync is an alternative to SCP that provides additional features like synchronization and compression.
Often times, it is more efficient for transferring large files or directories.

#### Note:
- It may be necessary to specify the path to the private key:

```bash
scp -i /path/to/private_key localfile username@hostname:/remote/directory
```
or

```bash
rsync -avz -e "ssh -i /path/to/private_key" localfile username@hostname:/remote/directory
```

### 3. **Running Commands on a Remote Server:**

```bash
ssh username@hostname 'command_to_run'
```

#### Explanation:
- This allows you to execute a command on the remote server without opening an interactive shell.

### 4. **SSH Key Generation:**

```bash
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
```

#### Explanation:
- `ssh-keygen`: Command to generate a new SSH key pair.
- `-t rsa`: Specifies the type of key to create (RSA).
- `-b 4096`: Specifies the number of bits in the key (4096 bits for stronger security).
- `-C "your_email@example.com"`: Adds a comment to the key (typically your email address).

### 5. **Adding SSH Key to the SSH Agent:**

```bash
eval "$(ssh-agent -s)"
ssh-add /path/to/private_key
```
#### Explanation:
- `eval "$(ssh-agent -s)"`: Starts the SSH agent in the background.
- `ssh-add`: Adds your private key to the SSH agent for easier authentication.

### 6. **SSH Config File:**
You can create a config file to simplify SSH commands:

```bash
~/.ssh/config
```

#### Example Config:
```
Host myserver
HostName hostname
User username
IdentityFile /path/to/private_key
```

#### Explanation:
- This allows you to connect to the server using a simple command:
```bash
ssh myserver
```

### 7. **Slurm and SSH:**

When connected to a cluster using SSH, you may need to submit jobs using Slurm. To get an interactive session on a compute node, you can use:
```bash
srun -p interactive --gres=gpu:1 --pty --time=9:59:00 bash
```

#### Explanation:
- `srun`: Command to submit a job to the Slurm scheduler.
- `-p interactive`: Specifies the partition to use (interactive in this case).
- `--gres=gpu:1`: Requests one GPU for the job.
- `--pty`: Allocates a pseudo-terminal for interactive use.
- `--time=9:59:00`: Sets a time limit for the job (9 hours and 59 minutes in this case).
- `bash`: Specifies the command to run (in this case, a bash shell).

### 8. **Module Management with SSH:**

When working on a cluster, you may need to load specific modules for your environment. You can do this after connecting via SSH:

```bash
module load module_name
```

To see available modules, you can use:

```bash
module avail
```

#### Explanation:
- `module load`: Command to load a specific module into your environment.
- `module avail`: Command to list all available modules on the cluster.

### 9. **Terminal Multiplexing with tmux:**

To keep your SSH sessions alive and manage multiple terminal windows, you can use `tmux`:

```bash
tmux
```

or, to start a new session with a name:

```bash
tmux new -s session_name
```

#### Explanation:
- `tmux`: Terminal multiplexer that allows you to create, manage, and navigate between multiple terminal sessions within a single SSH connection.
- You can detach from a tmux session with `Ctrl + b` followed by `d`, and reattach later with `tmux attach`. To close a tmux session, typing `exit` within the session will work.
- To list all tmux sessions, you can use `tmux ls`.
- To attach to a specific session, use `tmux attach -t session_name`.
- This is particularly useful for long-running processes or when you want to maintain multiple tasks in parallel without needing multiple SSH connections.
