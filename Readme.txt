Readme depthmap

Compile:
Depthmap requires the OpenCV library and a compatible Nvidia
graphics card. OpenCV must be compiled with Cuda-support. 
The included sources have been tested with OpenCV-library 
versions 2.4.10 and 3.1 on Windows (Visual Studio 2013, Cuda 6.5). To compile, 
first build  and install OpenCV, then edit the included
Makefile to reflect your setup, and run 'nmake' from the VisualStudio command prompt.

The sources should compile with little changes on MacOS and Linux-targets
as well.

Install:
To run the supplied depthmap-executable, you need to manually install
OpenCV-3.10 binaries with Cuda-support. Unfortunately, the official distribution
does not include Cuda; you might try one of the inofficial distributions
like the one from nuget.org, which requires the additional installation of
Cuda 7.5. I could not test this version since my graphics card only works with
Cuda 6.5.


Usage:
See the included manpage and the program sources for details. The example
images were processed using the command
depthmap -r 0,8,1967,1116 -ox depth.png -mn -3 -mx 23 img1.jpg img2.jpg
