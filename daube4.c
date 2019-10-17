/****************************************************************************************
*                       HOMEWORK4: Wavelet Image Compression using hybrid MPI/OpenMP    *
*                       STUDENT: Hoa Trinh                                              *
*****************************************************************************************/
#include <stdio.h>
#include "mpi.h"
#include <omp.h>
#define N 512					/* # of pixels in the orginal image */
#define L 3					/* # of recursive halvings */
#define MAXLINE 1024				/* ??? */
#define MAX 255 				/* max intensity */
#define C0 0.482962
#define C1 0.836516
#define C2 0.224143
#define C3 -0.129409
	
int vproc[2]={2,2};				/* 2x2 spatial decomposition */
int nproc = 4;					/* # of MPI ranks = vproc[0](row) x vproc[1](column) */
int nr, nc;					/* # of row and cols per rank */
int sid;                                        /* Rank of an MPI process */
int nthread=4; 					/* # of OpenMP threads */
double img[N+2][N+2];
double sbuf[N*N/4], rbuf[N*N/4];		/* sent and received buffers */

void copy(int row, int col, int sr, int sc)
{
	int i=0;
	int r,c;

        for (r=0; r<row; r++) {
                        for (c=0; c<col; c++) {
				sbuf[i]=img[r+sr*nr][c+sc*nc];
                                i++;
                        }
	}

}

void paste(int row, int col, int sr, int sc)
{
	int i=0;
	int r,c;

	for (r=0; r<row; r++) {
                        for (c=0; c<col; c++) {
                                img[r+sr*nr][c+sc*nc] = rbuf[i];
                                i++;
                        }
	}

}

void read_img()
{
	int r,c,n,s,i,sr,sc;
	FILE *f;
	char line[MAXLINE];
	
	MPI_Status status;

    	if (sid==0) {
		f=fopen("Lenna512x512.pgm","r");
  		fgets(line,MAXLINE,f);
		fgets(line,MAXLINE,f);
		fgets(line,MAXLINE,f);
		sscanf(line,"%d %d",&nc,&nr);
		fgets(line,MAXLINE,f);
		sscanf(line,"%d",&n);

		for (r=0; r<nr; r++) 
    			for (c=0; c<nc; c++) 
				img[r][c] = (double)fgetc(f);
    		
  		fclose(f);
    	}

	/* Initial */
	nr = N/vproc[0];
	nc = N/vproc[1];

    	for (s=1; s<nproc; s++) {
		sr = s/vproc[1];
		sc = s%vproc[1];

		if (sid==0) {
			copy(nr, nc, sr, sc);
			MPI_Send(&sbuf[0], nr*nc, MPI_DOUBLE, s, s, MPI_COMM_WORLD);
		}
		else if (sid==s) {
			MPI_Recv(&rbuf[0], nr*nc, MPI_DOUBLE, 0, sid, MPI_COMM_WORLD, &status);
                	paste(nr, nc, 0, 0);
		}
    	}
}

void wavelet()
{
	int r,c,tid;
	int S0, S1;
	int vid[2];                        		/* Vector process ID (rank) */
	int nbr[2];                       		/* Neighbor process ID in row & cols directions*/

	MPI_Request request;
	MPI_Status status;

	/* OpenMP setup */
        omp_set_num_threads(nthread);

        vid[0] = sid/vproc[1];
        vid[1] = sid%vproc[1];

        S0 = (vid[0]-1+vproc[0])%vproc[0];
        S1 = vid[1];
        nbr[0] = S0*vproc[1]+S1;                	/* row neighbor */

        S0 = vid[0];
        S1 = (vid[1]-1+vproc[1])%vproc[1];
        nbr[1] = S0*vproc[1]+S1;                	/* column neighbor */

	/* copy the first 2 rows to send to a row neighbor */
	copy(2, nc, 0, 0);			
	MPI_Irecv(&rbuf[0], 2*nc, MPI_DOUBLE, MPI_ANY_SOURCE, 20, MPI_COMM_WORLD, &request); 
	MPI_Send(&sbuf[0], 2*nc, MPI_DOUBLE, nbr[0], 20, MPI_COMM_WORLD);
	MPI_Wait(&request,&status);
	/* receive buffer and append it to the last 2 rows */ 
	paste(2, nc, 1, 0);

	#pragma omp parallel private(r, tid, c)
	{
        	tid = omp_get_thread_num();
		/*if set r = tid, result is wrong since threads will rewrite img (note: 2*r,2r+1...) */
		for (c=tid; c<nc; c+=nthread) {
			for (r=0; r<nr/2; r++) {
				img[r][c] = C0*img[2*r][c]+C1*img[2*r+1][c]+C2*img[2*r+2][c]+C3*img[2*r+3][c];
			}
			
		}
	}
	
	/* copy the first 2 cols to send to a column neighbor */
        copy(nr/2, 2, 0, 0);
	MPI_Irecv(&rbuf[0], nr, MPI_DOUBLE, MPI_ANY_SOURCE, 30, MPI_COMM_WORLD, &request);
        MPI_Send(&sbuf[0], nr, MPI_DOUBLE, nbr[1], 30, MPI_COMM_WORLD);
        MPI_Wait(&request,&status);
	/* receive buffer and append it to the last 2 cols */
	paste(nr/2, 2, 0, 1);

	
	#pragma omp parallel private(tid, r, c)
        {
                tid = omp_get_thread_num();

		for (r=tid; r<nr/2; r+=nthread) {
                        for (c=0; c<nc/2; c++) {
                                img[r][c] = C0*img[r][2*c]+C1*img[r][2*c+1]+C2*img[r][2*c+2]+C3*img[r][2*c+3];
                        }
                }
	}
}

void write_img()
{
	int s, sr, sc;
	int r, c;
	double max, resize;
	double work;
	MPI_Status status;

	for (s=1; s<nproc; s++) {
		sr = s/vproc[1];
		sc = s%vproc[1];
		if (sid==s) {
			copy(nr, nc, 0, 0);
			MPI_Send(&sbuf[0], nr*nc, MPI_DOUBLE, 0, s, MPI_COMM_WORLD);
		}
		else if (sid==0) {
			MPI_Recv(&rbuf[0], nr*nc, MPI_DOUBLE, s, s, MPI_COMM_WORLD, &status);
			paste(nr, nc, sr, sc);
		}
	}

	if (sid==0) {
		/* rank 0 write compressed image */
		char filename[64];
	        FILE *f;
	        sprintf(filename, "lenna.pgm");
	        f = fopen( filename, "w");

		nc = vproc[1]*nc;
		nr = vproc[0]*nr;
	
	        fprintf(f,"P5\n");
	        fprintf(f,"# Wavelet compression \n");
	        fprintf(f,"%d %d\n",nc,nr);
	        fprintf(f,"%d\n",MAX);

		/* Find maximum element of matrix */
		max = img[0][0];
		for (r=0; r<nr; r++) {
			for (c=0; c<nc; c++) {
				if (img[r][c]>max) max = img[r][c];
			}
		}

		resize = max/255; 
	
	        for (r=0; r<nr; r++) {
	                for (c=0; c<nc; c++) {
	                        work=img[r][c]/resize; 		/* rescale image to [0,255] and write image */
	                        fputc((char)work,f);
	                }
	        }
	        fclose(f);
	}
}


int main(int argc, char *argv[])
{
	int l;
	MPI_Status status;
        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &sid);
        MPI_Comm_size(MPI_COMM_WORLD, &nproc);

	read_img();

	/* Recursive wavelet transforms */
	for (l=0; l<2; l++) {
		wavelet();
		nr /= 2;
		nc /= 2;
	}

	write_img();
	MPI_Finalize();
	return 0;
}



	

	

	
