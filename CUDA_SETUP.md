## CUDA Installation

- As mentioned in the [README](README.md), one of the easiest methods to get the framework up and running is probably with Conda -- you can use either [Miniconda](https://docs.anaconda.com/miniconda/) or [Anaconda](https://www.anaconda.com/). 

- Here's a quick method to install Miniconda on Linux:

1. **Download the Miniconda installer**:
   ```bash
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   ```

2. **Make the installer executable**:
   ```bash
   chmod +x Miniconda3-latest-Linux-x86_64.sh
   ```

3. **Run the installer**:
   ```bash
   ./Miniconda3-latest-Linux-x86_64.sh
   ```

   Follow the prompts and choose the installation directory.

4. **Initialize conda** (this adds conda to your shell’s startup):
   ```bash
   source ~/miniconda3/bin/activate
   conda init
   ```

5. **Restart your terminal or run**:
   ```bash
   source ~/.bashrc
   ```

- After that, Miniconda should be ready to use.

- I recommend installing `mamba` and using it instead of `conda` commands for way faster usability, like so:

  ```bash
  conda install mamba
  mamba create yolov8_env  
  mamba activate yolov8_env
  ```

- You can then install the required packages i.e. like this:

  ```bash
  mamba install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
  mamba install -c conda-forge ffmpeg
  ```

- (Note: the `pytorch-cuda` version can be newer than the one mentioned above, I have tested the program in various setups up to CUDA 12.4 and CUDA 12.6 -- however, given the convoluted nature of the toolkit, newer version might mean new issues.)

- If you're running multiple projects on the same machine, I highly recommend prioritizing the `conda`/`mamba` environment libraries instead of your regular `LD_LIBRARY_PATH`. You can check out your `LD_LIBRARY_PATH` with:

  ```bash
  echo $LD_LIBRARY_PATH
  ```

- Note that the `echo` command should show your `conda` libraries first, i.e.: `$HOME/miniconda3/envs/yolov8_env`.

- If your `conda` environment specific libraries aren't showing up at first, you can use the following commands to create an activation script that sets LD_LIBRARY_PATH to include only the Conda environment's `lib` directory, assuming you are using `miniconda3`:

  ```bash
  export ENV_NAME="yolov8_env"

  # Create the activation directory if it doesn't exist
  mkdir -p $HOME/miniconda3/envs/$ENV_NAME/etc/conda/activate.d

  # Create the activation script to set LD_LIBRARY_PATH
  echo 'export LD_LIBRARY_PATH=$HOME/miniconda3/envs/'$ENV_NAME'/lib' > $HOME/miniconda3/envs/$ENV_NAME/etc/conda/activate.d/env_vars.sh

  # Make the script executable
  chmod +x $HOME/miniconda3/envs/$ENV_NAME/etc/conda/activate.d/env_vars.sh
  ```

- If you're still missing libraries such as `cudnn`, **although not recommended in many use cases**, you _can_ also install CUDA related packages via `conda`/`mamba` by using Anaconda.org's `nvidia` [channel](https://anaconda.org/nvidia/), like so:

  ```bash
  mamba install -c nvidia cudatoolkit cudnn
  ```

- However, keep in mind that you should only do these installs inside a very case-specific `conda` environment, like the one created for the project, and that multiple overlapping installations may introduce problems. 

- Also, some users and developers have reported that the precompiled builds offered by Nvidia's channel aren't necessarily always compiled with all the bells and whistles intact.

- With all this said, good luck.