#include <iostream>
#include <fstream>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#if CV_MAJOR_VERSION < 3
	#include "opencv2/gpu/gpu.hpp"
#else
	#include "opencv2/cudaoptflow.hpp"
	#include "opencv2/cudaarithm.hpp"
#endif

using namespace std;
using namespace cv;
#if CV_MAJOR_VERSION < 3
	using namespace cv::gpu;
#else
	using namespace cv::cuda;
#endif


// Helper functions for command line handling

typedef struct{
	int n;
	char **s;
}list;

// create list of strings
list *list_create(int n, char **s){
	list* r = new list;
	r->n = n;
	r->s = (char**)malloc(n*sizeof(char*));
	memcpy(r->s,s,n*sizeof(char*));
	return r;
}

// get index of string s in list
int list_get(list *r, const char* s){
	int k;
	for(k=0; k<r->n; k++)
		if(!strcmp(s, r->s[k]))
			return k;
	return -1;
}

// remove string with index k from list
void list_remove(list *r, int k){
	if(k>=0 && k<r->n){
		memmove(r->s+k, r->s+k+1,(r->n-(k+1))*sizeof(char*));
		r->n--;
	}
}

// parse "2,4,2,1,3,56" into int-array
// length in element 0
int* parse_int_array( char *s){
	// how many ',' ?
	char *c,*s1=s;
	int nc = 0;
	while((c =strchr(s1,','))!=NULL){
		s1 = c+1;
		nc++;
	}
	int *x  = new int[nc+2],
		idx = 0;
	x[idx++] = nc+1;

	while((c=strchr(s,','))!=NULL){
		*c = '\0';
		x[idx++] = atoi(s);
		s = c+1;
	}
	x[idx++] = atoi(s);
	return x;
}
	

// get maximum of float-image
float max(Mat m){
	float mx = m.at<float>(0,0);
	for(int y=0; y<m.rows; y++){
		const float* fr = m.ptr<float>(y);
		for(int x = 0; x<m.cols; x++){
			float fx = fr[x];
			if(fx > mx)
				mx = fx;
		}
	}
    return mx;
}

// get minimum of float image
float min(Mat m){
	float mx = m.at<float>(0,0);
	for(int y=0; y<m.rows; y++){
		const float* fr = m.ptr<float>(y);
		for(int x = 0; x<m.cols; x++){
			float fx = fr[x];
			if(fx < mx)
				mx = fx;
		}
	}
    return mx;
}

// create alpha channel
void setalpha(Mat m, Rect *r){
	Point pt;
	for(pt.y=0; pt.y<m.rows; pt.y++){
		unsigned char* cr = m.ptr<unsigned char>(pt.y);
		for(pt.x = 0; pt.x<m.cols; pt.x++){
			cr[pt.x] = ( r->contains(pt) ? (unsigned char)255 : 0 ) ;
		}
	}
}


double *polyfit(double *x, double *y, double *p){
	double dx = x[1]-x[0];
	if(dx != 0){
		p[0] = (y[1]-y[0]) / dx,
		p[1] =  y[1]-p[0]*x[1];
		return p;
	}
	return NULL;
}


void usage(){
	cerr << "Usage: depthmap " 
						<< "-mx maximum "
						<< "-mn minimum "
						<< "-i [inverse] "
						<< "-v [verbose] "						
						<< "-r x,y,width,height "
						<< "-ox filex "
						<< "-oy filey "
						<< "-brox_alpha 0.197 "
						<< "-brox_gamma 80 "
						<< "-brox_scale 0.8 "
						<< "-brox_inner 10 "
						<< "-brox_outer 77 "
						<< "-brox_solver 10 "
						<< "<frame1> <frame2>" << endl;
	exit(0);
}


int main(int argc, char* argv[]){
//	cout << cv::getBuildInformation() << endl;
	auto idxc = getCudaEnabledDeviceCount();
	if (idxc == 0){
		cerr << "No Cuda device found." <<endl;
		return -1;
	}
		
	float mx=FLT_MAX,			// maximum flow
		  mn=FLT_MIN;			// minimum flow
	Rect *rct=NULL;				// roi
	char *foutx = NULL,			// outputfile for flow x
		 *fouty = NULL;			// outputfile for flow y
	bool inverse = false,		// close objects white
		 verbose = false;
	int  width, height;			// dimension of first source image = dimension of output
	
	// Brox Parameters 
	
    float alpha = 0.197f;		//  flow smoothness   
    float gamma = 50.0f;		//  gradient constancy importance
    float scale_factor = 0.8f;	//	pyramid scale factor
	int inner_iterations = 10;	//  number of lagged non-linearity iterations (inner loop)    
    int outer_iterations = 77;	//	number of warping iterations (number of pyramid levels)  
    int solver_iterations = 10; // 	number of linear system solver iterations
	
	// parse command line
	list *args = list_create(argc, argv);
	int idx;
	
	if( (idx=list_get(args,"-brox_alpha")) >= 0){  
		if(idx<args->n-1) {
			alpha = (float)atof(args->s[idx+1]);
			list_remove(args,idx);
			list_remove(args,idx);
		}else
			usage();
	}
	if( (idx=list_get(args,"-brox_gamma")) >= 0){  
		if(idx<args->n-1) {
			gamma = (float)atof(args->s[idx+1]);
			list_remove(args,idx);
			list_remove(args,idx);
		}else
			usage();
	}
	if( (idx=list_get(args,"-brox_scale")) >= 0){  
		if(idx<args->n-1) {
			scale_factor = (float)atof(args->s[idx+1]);
			list_remove(args,idx);
			list_remove(args,idx);
		}else
			usage();
	}
	if( (idx=list_get(args,"-brox_inner")) >= 0){  
		if(idx<args->n-1) {
			inner_iterations = atoi(args->s[idx+1]);
			list_remove(args,idx);
			list_remove(args,idx);
		}else
			usage();
	}
	if( (idx=list_get(args,"-brox_outer")) >= 0){  
		if(idx<args->n-1) {
			outer_iterations = atoi(args->s[idx+1]);
			list_remove(args,idx);
			list_remove(args,idx);
		}else
			usage();
	}
	if( (idx=list_get(args,"-brox_solver")) >= 0){  
		if(idx<args->n-1) {
			solver_iterations = atoi(args->s[idx+1]);
			list_remove(args,idx);
			list_remove(args,idx);
		}else
			usage();
	}
	if( (idx=list_get(args,"-ox")) >= 0){  
		if(idx<args->n-1) {
			foutx = args->s[idx+1];
			list_remove(args,idx);
			list_remove(args,idx);
		}else
			usage();
	}
	if( (idx=list_get(args,"-oy")) >= 0){  
		if(idx<args->n-1) {
			fouty = args->s[idx+1];
			list_remove(args,idx);
			list_remove(args,idx);
		}else
			usage();
	}
	if( (idx=list_get(args,"-mx")) >= 0){  
		if(idx<args->n-1) {
			mx =  (float)atof( args->s[idx+1] );
			list_remove(args,idx);
			list_remove(args,idx);
		}else
			usage();
	}
	if( (idx=list_get(args,"-mn")) >= 0){  
		if(idx<args->n-1) {
			mn =  (float)atof( args->s[idx+1] );
			list_remove(args,idx);
			list_remove(args,idx);
		}else
			usage();
	}
	if( (idx=list_get(args,"-r")) >= 0){  
		if(idx<args->n-1) {
			int *rp = parse_int_array( args->s[idx+1] );
			if(rp[0]!=4) // length
				usage();
			rct = &Rect(rp[1], rp[2], rp[3], rp[4]);
			list_remove(args,idx);
			list_remove(args,idx);
		}else
			usage();
	}
	if( (idx=list_get(args,"-i")) >= 0){  
		inverse = true;
		list_remove(args,idx);
	}
	if( (idx=list_get(args,"-v")) >= 0){  
		verbose = true;
		list_remove(args,idx);
	}

	if(args->n <3)
		usage();
	
	if(verbose){
		printCudaDeviceInfo(0);
	}
	
	// read input images 
	
    Mat frame0 = imread(args->s[1], IMREAD_GRAYSCALE);
    Mat frame1 = imread(args->s[2], IMREAD_GRAYSCALE);

    if (frame0.empty()){
        cerr << "Can't open image ["  << args->s[1] << "]" << endl;
        return -1;
    }
    if (frame1.empty()){
        cerr << "Can't open image ["  << args->s[2] << "]" << endl;
        return -1;
    }

	width 	= frame0.cols;
	height 	= frame0.rows;
	
	if(rct != NULL){
        frame0 = Mat(frame0, *rct);
        frame1 = Mat(frame1, *rct);
    }

    if (frame1.size() != frame0.size()){
        cerr << "Images should be of equal sizes" << endl;
        return -1;
    }

    GpuMat d_frame0(frame0);
    GpuMat d_frame1(frame1); 
 
    GpuMat d_frame0f;
    GpuMat d_frame1f;

    d_frame0.convertTo(d_frame0f, CV_32F, 1.0 / 255.0);
    d_frame1.convertTo(d_frame1f, CV_32F, 1.0 / 255.0);

    const int64 start = getTickCount();
		
#if CV_MAJOR_VERSION < 3
	GpuMat d_flowx(frame0.size(), CV_32FC1);
	GpuMat d_flowy(frame0.size(), CV_32FC1);

	BroxOpticalFlow brox(alpha, gamma, scale_factor, inner_iterations, outer_iterations, solver_iterations);
	brox(d_frame0f, d_frame1f, d_flowx, d_flowy);
 
	Mat flowx(d_flowx);
	Mat flowy(d_flowy);
#else
	GpuMat d_flow(frame0.size(), CV_32FC2);
    Ptr<cuda::BroxOpticalFlow> brox = cuda::BroxOpticalFlow::create(alpha, gamma, scale_factor, inner_iterations, outer_iterations, solver_iterations);
	brox->calc(d_frame0f, d_frame1f, d_flow);
	GpuMat planes[2];
	cuda::split(d_flow, planes);

	Mat flowx(planes[0]);
	Mat flowy(planes[1]);
#endif
	
    const double timeSec = (getTickCount() - start) / getTickFrequency();
    cout << "Brox : " << timeSec << " sec" << endl;
		
	// optical flow x
		
	float mxx = max(flowx),
		  mnx = min(flowx);

	cerr << "Flow x: Max " << mxx << " Min " << mnx << endl;
		
	if(foutx != NULL) {
		mxx = (mx==FLT_MAX?mxx:mx);
		mnx = (mn==FLT_MIN?mnx:mn);
		double p[2], 
			   x[] {inverse?mnx:mxx,inverse?mxx:mnx}, 
			   y[] {0,255} ;
		polyfit( x, y, p );
		flowx  = (float)p[0]*flowx+(float)p[1];

		Mat depth;			
		flowx.convertTo(depth, CV_8UC1);
		if(rct != NULL){  // enlarge and add alpha channel
			Mat alpha(height, width, CV_8UC1);
			setalpha(alpha,rct);
			Mat data(height, width, CV_8UC1);
			depth.copyTo(data.colRange(rct->x,rct->x+rct->width).rowRange(rct->y,rct->y+rct->height));
			
			vector<Mat> tmp;
			tmp.push_back(data);
			tmp.push_back(data);
			tmp.push_back(data);
			tmp.push_back(alpha);
			merge(tmp, depth);
		}		
		imwrite(foutx,depth);
	}
			  
	// optical flow y
		
	float mxy = max(flowy),
		  mny = min(flowy);

	cerr << "Flow y: Max " << mxy << " Min " << mny << endl;
		
	if(fouty != NULL) {
		mxy = (mx==FLT_MAX?mxy:mx);
		mny = (mn==FLT_MIN?mny:mn);
		double p[2], 
			   x[] {inverse?mny:mxy,inverse?mxy:mny}, 
			   y[] {0,255} ;
		polyfit(x,y, p);
		flowy = (float)p[0]*flowy+p[1];

		Mat depth;			
		flowy.convertTo(depth, CV_8UC1);
		if(rct != NULL){  // enlarge and add alpha channel
			Mat alpha(height, width, CV_8UC1);
			setalpha(alpha,rct);
			Mat data(height, width, CV_8UC1);
			depth.copyTo(data.colRange(rct->x,rct->x+rct->width).rowRange(rct->y,rct->y+rct->height));
				
			vector<Mat> tmp;
			tmp.push_back(data);
			tmp.push_back(data);
			tmp.push_back(data);
			tmp.push_back(alpha);
			merge(tmp, depth);
		}		
		imwrite(fouty,depth);
	}
			  
   
    return 0;
}
