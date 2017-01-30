CC		=	cl.exe
COPT	=	/EHsc /O2 /MD
CINC241	=	/I"C:\Users\dersch\dev\opencv\vcbuild\install\include" 

CINC	=	/I"C:\Users\dersch\dev\opencv310\build2\install\include" 

LINKER	= 	link.exe
LIBS241	=	"C:\Users\dersch\dev\opencv\vcbuild\install\x64\vc12\lib\opencv_core2410.lib" \
			"C:\Users\dersch\dev\opencv\vcbuild\install\x64\vc12\lib\opencv_gpu2410.lib" \
			"C:\Users\dersch\dev\opencv\vcbuild\install\x64\vc12\lib\opencv_highgui2410.lib" 

LIBS	=	"C:\Users\dersch\dev\opencv310\build2\install\x64\vc12\lib\opencv_core310.lib" \
			"C:\Users\dersch\dev\opencv310\build2\install\x64\vc12\lib\opencv_cudaarithm310.lib" \
			"C:\Users\dersch\dev\opencv310\build2\install\x64\vc12\lib\opencv_cudaoptflow310.lib" \
			"C:\Users\dersch\dev\opencv310\build2\install\x64\vc12\lib\opencv_highgui310.lib" \
			"C:\Users\dersch\dev\opencv310\build2\install\x64\vc12\lib\opencv_imgcodecs310.lib" 


OBJ		=	depth_map.o



depthmap.exe	: $(OBJ) Makefile
	$(LINKER) $(OBJ) $(LIBS) /OUT:depthmap.exe


depth_map.o	:	depth_map.cpp Makefile
	$(CC) $(CINC) $(COPT)  /c depth_map.cpp /Fo:depth_map.o


			
	