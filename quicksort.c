/****************************************************************************************
* 			HOMEWORK3: Hypercube Quicksort using MPI			*
* 			STUDENT: Hoa Trinh						*
*****************************************************************************************/
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MAX 99			/* Maximum value of a list element */
#define N 256			/* Maximum list size : 64 core, 4 elements at each core */
#define MAXD 6			/* The largest hypercube dimension */
#define MAXP 64			/* Maximum number of core 2^6 */

int nprocs,dim,myid; 		/* Cube size, dimension, &my rank */
int n=4; 			/* initial number of list elements at a processor */

/******************************* Sequential Quicksort ***********************************
*		Input: A list, indices of the first and the last element. 		*
*****************************************************************************************/
void quicksort(int list[],int left,int right) {
	int pivot,i,j;
	int temp;

	if (left < right) {
		i = left; j = right + 1;
		pivot = list[left];
		do {
			while (list[++i] < pivot && i <= right);
			while (list[--j] > pivot);
			if (i < j) {
				temp = list[i]; list[i] = list[j]; list[j] = temp;
			}
		} while (i < j);
		temp = list[left]; list[left] = list[j]; list[j] = temp;
		quicksort(list,left,j-1);
		quicksort(list,j+1,right);
	}
}

void parallel_quicksort(int myid, int list[])
{
	int mask;					/* To choose a subcube master */
	int bitvalue;					/* To devive a subcube into 2 parts */
	int L, p, c, nprocs_cube;
	int count;					/* total elements of a subcube */
	int nsend, nrecv;	 			/* number of send and receive elements */
	int left[MAXD][N], right[MAXD][N];		/* left and right sublists */
	int listsum; 					/* sum of all list elements */
	int partner;
	int total, pivot; 				/* sum and average of all elements within a subcube */
	int i, j;					/* number of elements not greater and greater than a pivot */
	int procs_cube[nprocs_cube];
	MPI_Status status;

	bitvalue = nprocs >> 1;				/* 100 -> 010 -> 001; right shift or devide by 2 */
	mask = nprocs - 1;
	nprocs_cube = nprocs;
	MPI_Comm cube[MAXD][MAXP];	             	/* Communicators within each subcube */
        MPI_Group cube_group[MAXD][MAXP];      		/* Subcube */
	MPI_Comm_size(MPI_COMM_WORLD,&nprocs_cube);
        cube[dim][0] = MPI_COMM_WORLD;

	for (L=dim; L>=1; L--) {
		i = 0;
		j = 0;
		listsum = 0;
		c = myid/nprocs_cube;

		// Master subcube pick a pivot value
		if ((myid & mask) == 0){
			for (p = 0; p<n; p++) listsum += list[p];
			if (n>0) pivot = listsum/n;
		}

		// Broadcast the pivot from the master to other members of the subcube
		MPI_Bcast(&pivot, 1, MPI_INT, 0, cube[L][c]);
		
		//Partition list into 2 sublists
		for (p=0; p<n; p++) {
			if (list[p] <= pivot){
				left[L][i] = list[p];
				i += 1;
			} else {
				right[L][j] = list[p];
				j += 1;
			}
		}
		
		partner = myid ^ bitvalue; 
		if ((myid & bitvalue) == 0) {
			nsend = j;
			MPI_Send(&nsend, 1, MPI_INT, partner, myid, MPI_COMM_WORLD);

			//only send or receive if the listsize is nonzero
			if (nsend > 0) {
				MPI_Send(&right[L][0], j, MPI_INT, partner, myid, MPI_COMM_WORLD);
			}

			MPI_Recv(&nrecv, 1, MPI_INT, partner, partner, MPI_COMM_WORLD, &status);
			if (nrecv > 0) {
				MPI_Recv(&left[L][i], nrecv, MPI_INT, partner, partner, MPI_COMM_WORLD, &status);
			}
			n = n - nsend + nrecv;

			//Change list to left list
	                for (p=0; p<n; p++) list[p]= left[L][p];

		} else {
			nrecv = i;
			MPI_Send(&nrecv, 1, MPI_INT, partner, myid, MPI_COMM_WORLD);

			//only send or receive if the listsize is nonzero
			if (nrecv > 0) {
                        	MPI_Send(&left[L][0], i, MPI_INT, partner, myid, MPI_COMM_WORLD);
			}
			MPI_Recv(&nsend, 1, MPI_INT, partner, partner, MPI_COMM_WORLD, &status);
			if (nsend > 0) {
                        	MPI_Recv(&right[L][j], nsend, MPI_INT, partner, partner, MPI_COMM_WORLD, &status);
			}
			n = n + nsend - nrecv;

                        //Change list to right list
                        for (p=0; p<n; p++) list[p]= right[L][p];

		}
		
		// At each level
		MPI_Comm_group(cube[L][c],&(cube_group[L][c]));
		nprocs_cube = nprocs_cube/2;
		
		for(p=0; p<nprocs_cube; p++) procs_cube[p] = p;
		MPI_Group_incl(cube_group[L][c],nprocs_cube,procs_cube,&(cube_group[L-1][2*c ]));
		MPI_Group_excl(cube_group[L][c],nprocs_cube,procs_cube,&(cube_group[L-1][2*c+1]));
		MPI_Comm_create(cube[L][c],cube_group[L-1][2*c ],&(cube[L-1][2*c ]));
		MPI_Comm_create(cube[L][c],cube_group[L-1][2*c+1],&(cube[L-1][2*c+1]));
		
		MPI_Group_free(&(cube_group[L ][c ]));
		MPI_Group_free(&(cube_group[L-1][2*c ]));
		MPI_Group_free(&(cube_group[L-1][2*c+1]));		

		mask ^= bitvalue;
		bitvalue /= 2;

	}
			
	quicksort(list, 0, n-1);
}	


int main(int argc, char *argv[])
{
	int list[N];
	int k;
	
	MPI_Status status;
	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&myid);
	MPI_Comm_size(MPI_COMM_WORLD,&nprocs);

	dim = log(nprocs+1e-10)/log(2.0);

	/* Initially, each process has 4 elements generated randomly */
	srand((unsigned) myid+1);
	for (k=0; k<n; k++) list[k] = rand()%MAX;

	printf("Before: Rank %2d:", myid);
        for (k=0; k<n; k++) printf("%3d", list[k]);
        printf("\n");

	parallel_quicksort(myid, list);
	printf("After: Rank %2d:", myid);
        for (k=0; k<n; k++) printf("%3d", list[k]);
        printf("\n");

	MPI_Finalize();

	return 0;
}


		

