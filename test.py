import subprocess
import os

jags_path = r"C:/Program Files/JAGS/JAGS-4.3.1/x64/bin/jags.bat"
workdir = r"C:/Users/PC/Downloads/JAGS"

# 1. Write model
with open(os.path.join(workdir, "model.bug"), "w") as f:
    f.write("""
model {
  x <- 3
  tau <- 1/(5*5)  # y=5, convert to precision
  z ~ dnorm(x, tau)
}
""")

# 2. Write script
with open(os.path.join(workdir, "script.txt"), "w") as f:
    f.write("""
model in "model.bug"
compile, nchains(1)
initialize
update 1000
monitor z
update 1000
coda *
""")

# 3. Run JAGS
cmd = f'"{jags_path}" script.txt'
output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True, cwd=workdir).decode()

print(output)

# 4. Check CODA files
print("CODA files in folder:", os.listdir(workdir))
