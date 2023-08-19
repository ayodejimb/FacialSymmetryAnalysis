# FacialSymmetryAnalysis
GitHub Repository for Facial Symmetry Analysis Project

# The Folders
This repository contains a standalone executable of the project; you can find this in the 'standalone' folder, as well as the python script and the specification (.spec) file that was used in building the executable (these are enclosed in the 'Python_scripts_&_spec_file' folder). The python script (.py) file contains the implementation in Python. In the 'for_conversion' folder, you will find a '.txt' file to install the requirements on your machine in the case that you want to connvert the Python Script (.py) file to a standalone executable. 

# The Executable
The executable is developed to work with Windows Operating System. You don't need to install any dependency on your machine before launching. To launch the executable, double-click to open it. 
The OAK-D camera developed by Luxonis was configured with this project, ensure the camera is connected to your machine to starts the 'live capture' after launching the executable. 

# The .SPEC file and .py file
The project was developed in Python and the implemntation can be found in the '.py' file. The '.spec' file was used together with the '.py' file to convert the implementation into a standalone executable using pyInstaller. To do this, first navigate to the 'for conversion folder' through the terminal and run: 'pip install -r requirements.txt' . This will install the needed requirements on your machine. Before running the next command, it is important to change the directory in the '.spec' file and '.py' file to that of your local machine. Something like 'D:\\Desktop\\.....' This should be done for the 'facenet_pytorch' and 'fer' in the .spec file and 'Haar Cascade' in the '.py' file.  'Then type and run in terminal 'pyinstaller FacialSymmetryAnalysis.spec'. This will convert the script into a standalone executable. 

# Test Video
[Enclosed here](https://drive.google.com/file/d/1mIe6s983FV6RjA9Vsrs5Ns-MLv3G9VDR/view) is a test video of the implementation in the standalone executable. 
