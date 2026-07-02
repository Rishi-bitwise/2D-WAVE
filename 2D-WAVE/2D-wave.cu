#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>
#include<math.h>


#define SIDE 256                           //Actual dimension of the plate
#define TOTAL_TIME 1024                 //we will be skipping every 32 steps, from the kernel, but also we will record less to keep filesize small
#define TILESIZE 16
#define GRIDSIZE (SIDE/TILESIZE)
#define SHLOOPDIM 3                         //NUMBER of tiles in sharedmem, NEEDS TO BE ODD.
#define HALO2 (SHLOOPDIM/2)*TILESIZE                      //32 halos above, below left and right of the central tile
#define KERNEL_TIME_SKIPS 1
#define TOTAL_TIME_ROWS ((TOTAL_TIME)/(HALO2*KERNEL_TIME_SKIPS))
#define SHMEMDIM (SHLOOPDIM*TILESIZE)
#define PI 3.14159f
#define BOUNDARY_TEMP 0.0f                //actually boundary value, its a left over name from the heat pde
#define R 0.125f


/*This is heavily inspired by the 2D-HEAT PDE, so a lot of the concepts are repeated here. We have to take care of the spacial dimension
as well as the temporal dimension. The current value depends on its neighbours and its previous value. THe idea is to use the spacial technique
used in the heat-solver, and the temporal will simply be present in the global memory. The past value cannot be saved in shared memory, as we wont
have enough space for it, moreover there is no need to save it there, since we are not repeatedly accessing the same value, neither
are we trying to access any value column wise. So all that complexity is gone, and we fetch the past data from the global memory. Each block will
have a SHLOOPDIM X SHLOOPDIM size of GMEM for it to use as a scratchpad for the past values. It will read and write to that GMEM location */

/*NOTE We techinically dont need shbx and shby, since we now have scratchpad, and we can refactor the code to remove it, and depend on scratchpad, since
it has the old values. but for now, lets just make it work.*/
__global__ void init_temp(float *d_past, float *d_in)
{
    int xid = threadIdx.x + blockIdx.x*TILESIZE;
    int yid = threadIdx.y + blockIdx.y*TILESIZE;
    float time1 = 1;
    float time2 = 1.01;

    d_past[yid*SIDE + xid] = 100*sinf(2*PI*(time1*((float)(xid)+2*yid)/SIDE));
    d_in[yid*SIDE + xid] = 100*sinf(2*PI*(time2*((float)(xid)+2*yid)/SIDE));
}

/*d_scratch has dims : [GLOBAL BLOCKID/TILENO][SHMEMDIM][SHMEMDIM] ; so a specific tile has its own "shared memory" in global space. THis memory
is meant to be private to the particular TILE, although other tiles can access, but we will not do that. The idea being that this region
is going to be used as a scratch pad. We dont need a global shbx and shby. The reason we dont is because :
the scratchpad keeps the record of the tile's shmem's past values. now shmem get updated tile by tile. once a tile gets updated, its past value now
needs to go to the scratchpad. we can simply copy paste it there, without any issues. There is no question of prevx or prevy in the scratchpad.
We just fetch the values directly and write directly. just making sure it is coalesced.
*/
__global__ void run_sim(float *d_past, float *d_in, float *d_out, float *d_out_past, float *d_scratch)         //past and in are just 1 time step apart. similarly d_out and d_out_past are 1 time step apart
{
    int xid = threadIdx.x + blockIdx.x*TILESIZE;
    int yid = threadIdx.y + blockIdx.y*TILESIZE;

    int thx = threadIdx.x;
    int thy = threadIdx.y;

    __shared__ float shmem[SHMEMDIM+1][SHMEMDIM+1];                 // Padding is necessary because we need the columns as well.
    //SHMEMDIM is the 3x3 tile with halos.
    __shared__ float shbx[SHLOOPDIM][SHLOOPDIM-1][TILESIZE];           //this is for boundary between t0/t1 or t4/t5.
    __shared__ float shby[SHLOOPDIM-1][SHLOOPDIM][TILESIZE];
    //this is for boundary between t0/t3 or t5/t8. Note that the dimensions are transposed, because we wanna follow t0->t1->t2->t3.
    // that means that


    //--------------------------------------LOAD THE DATA FROM GMEM INTO SHMEM--------------------------------------//

    int globalxstart = blockIdx.x*TILESIZE - HALO2;
    int globalystart = blockIdx.y*TILESIZE - HALO2;

    for(int i=0 ;i<SHLOOPDIM; i++)                      //At SHLOOPDIM/2=HALO2 we are inside the global space, we are at the original tile
    {
        for(int j=0; j<SHLOOPDIM; j++)
        {
            int globalx = (globalxstart+j*TILESIZE+thx);
            int globaly = (globalystart+i*TILESIZE+thy);

            if (globalx<0 || globalx>=SIDE || globaly<0 || globaly>= SIDE)      //if the global index we are calculating with the tile at the center, is valid then we copy
                shmem[i*TILESIZE+thy][j*TILESIZE+thx] = BOUNDARY_TEMP;
            else
                shmem[i*TILESIZE+thy][j*TILESIZE+thx] = d_in[globaly*SIDE + globalx];
        }
    }

    __syncthreads();

    //--------------------------------------LOAD INTO D_SCRATCH FROM D_PAST--------------------------------------//

    int tileid = blockIdx.x + blockIdx.y*GRIDSIZE;          //Gives the acutal tile number, now all the threads in this tile will loop through the SHLOOPDIM X SHLOOPDIM
    for(int i=0; i<SHLOOPDIM; i++)
    {
        for(int j=0; j<SHLOOPDIM; j++)
        {
            int globalx = globalxstart + j*TILESIZE + thx;
            int globaly = globalystart + i*TILESIZE + thy;
            if (globalx<0 || globalx>=SIDE || globaly<0 || globaly>= SIDE)
                d_scratch[tileid*SHMEMDIM*SHMEMDIM + (i*TILESIZE+thy)*SHMEMDIM + j*TILESIZE+thx] = BOUNDARY_TEMP;
            else
                d_scratch[tileid*SHMEMDIM*SHMEMDIM + (i*TILESIZE+thy)*SHMEMDIM + j*TILESIZE+thx] = d_past[globaly*SIDE + globalx];
        }
    }


    //--------------------------------------START THE UPDATE LOOP TO GO OVER THE TILES IN THE SHMEM--------------------------------------//

    for(int time=0; time<HALO2; time++)
    {
        //--------------------------------------LOAD THE DATA FROM SHMEM INTO SHBOUNDS--------------------------------------//

        /*
        shbx records : t0-t1, t1-t2, t3-t4, t6-t7, t7-t8. that is t0's last col, t1's last col, t3's last col, and so on.
        It is not that important to get the best performance, we could just manually loop through them and store them using only 1 col pf 32 threads.
        But we can take 2 col of threads, point them at t0-end and t1-end and then increment by 3 tiles for 3 iterations to get the data.
        */

        __syncthreads();
        if (thy<SHLOOPDIM-1)           //grab two rows of 32/16 threads. conditional branching penalty for thy only.
        {
            for(int i=0; i<SHLOOPDIM; i++)
            {
                shbx[i][thy][thx] = shmem[i*TILESIZE + thx][thy*TILESIZE + TILESIZE-1];
                shby[thy][i][thx] = shmem[thy*TILESIZE + TILESIZE-1][i*TILESIZE+thx];           //Packing this here was good move. instead of another for/if
            }
        }
        __syncthreads();

        //--------------------------------------SHBOUNDS LOADED--------------------------------------//


        float prevx = 0;
        float prevy = 0;

        for(int i=0; i<SHLOOPDIM; i++)
        {
            for(int j=0; j<SHLOOPDIM; j++)
            {
                float curr = shmem[i*TILESIZE+thy][j*TILESIZE+thx];

                bool nextx_exists = (globalxstart + j*TILESIZE+thx+1)<SIDE;         //these are checks are most likely not necessary, need to check TODO.
                bool nexty_exists = (globalystart + i*TILESIZE+thy+1)<SIDE;
                bool prevx_exists = (globalxstart + j*TILESIZE+thx-1)>=0;
                bool prevy_exists = (globalystart + i*TILESIZE+thy-1)>=0;

                nextx_exists = nextx_exists && !(j==SHLOOPDIM-1 && thx==TILESIZE-1);                  //these are checks for outside shared memory
                nexty_exists = nexty_exists && !(i==SHLOOPDIM-1 && thy==TILESIZE-1);
                prevx_exists = prevx_exists && !(j==0 && thx==0);
                prevy_exists = prevy_exists && !(i==0 && thy==0);

                float nextx, nexty;
                if (nextx_exists)
                    nextx = shmem[i*TILESIZE+thy][j*TILESIZE+thx+1];       //*((int)(nextx_exists)) + BOUNDARY_TEMP*((int)(!nextx_exists));
                else
                    nextx = BOUNDARY_TEMP;

                if (nexty_exists)
                    nexty = shmem[i*TILESIZE+thy+1][j*TILESIZE+thx];       //*((int)(nexty_exists)) + BOUNDARY_TEMP*((int)(!nexty_exists));
                else
                    nexty = BOUNDARY_TEMP;

                //Unfortunately this is the [BUG FIX-1]. The previous tiles have already been updated. this is where shbx and shby shine!!!
                if (thx==0)                                 // HAS to be thx, because of the location within the tile
                {
                    if (j!=0)
                    {
                        if (prevx_exists)
                            prevx = shbx[i][j-1][thy];
                        else
                          prevx = BOUNDARY_TEMP;
                    }
                    else
                      prevx = BOUNDARY_TEMP;
                }
                else
                    prevx = shmem[i*TILESIZE+thy][j*TILESIZE+thx-1]*((int)(prevx_exists)) + BOUNDARY_TEMP*((int)(!prevx_exists));

                if (thy==0)                                         //can be written with bool, but readability takes a hit.
                {
                    if (i!=0)
                    {
                        if (prevy_exists)
                            prevy = shby[i-1][j][thx];
                        else
                          prevy = BOUNDARY_TEMP;
                    }
                    else
                      prevy = BOUNDARY_TEMP;
                }
                else
                    prevy = shmem[i*TILESIZE+thy-1][j*TILESIZE+thx]*((int)(prevy_exists)) + BOUNDARY_TEMP*((int)(!prevy_exists));

                float past_curr = d_scratch[tileid*SHMEMDIM*SHMEMDIM + (i*TILESIZE+thy)*SHMEMDIM + j*TILESIZE+thx];

                float new_curr = 2.0*curr - past_curr + R*(nextx + nexty + prevx + prevy - 4*curr);

                __syncthreads();

                shmem[i*TILESIZE+thy][j*TILESIZE+thx] = new_curr;
                d_scratch[tileid*SHMEMDIM*SHMEMDIM + (i*TILESIZE+thy)*SHMEMDIM + j*TILESIZE+thx] = curr;

                __threadfence_block();              //dscratch is in global mem, so we want all the warps to finish with all the tiles in dscratch

                __syncthreads();
            }
        }
    }

    d_out[yid*SIDE + xid] = shmem[HALO2+thy][HALO2+thx];
    d_out_past[yid*SIDE + xid] = d_scratch[tileid*SHMEMDIM *SHMEMDIM + (HALO2+thy)*SHMEMDIM + HALO2+thx];
}

int main()
{
    float *h_arr;                           // It needs to record the 2d plate(SIDE*SIDE) for TOTAL_TIME_ROWS times
    int platedim = SIDE*SIDE;
    h_arr = (float*) malloc(platedim*TOTAL_TIME_ROWS*sizeof(float));

    float *d_in ,*d_out, *d_past, *d_scratch, *d_out_past;
    float * tempo, *tempo2;
    cudaMalloc(&d_in, platedim*sizeof(float));
    cudaMalloc(&d_past, platedim*sizeof(float));
    cudaMalloc(&d_out, platedim*sizeof(float));
    cudaMalloc(&d_out_past, platedim*sizeof(float));
    cudaMalloc(&d_scratch, GRIDSIZE*GRIDSIZE * SHMEMDIM*SHMEMDIM *sizeof(float));                            //Each actural real tile, will have 8 halos, so totally 9 tiles, per tile.

    int gridlen = (SIDE+TILESIZE-1)/TILESIZE;

    dim3 blocksize(TILESIZE, TILESIZE);

    dim3 gridsize(gridlen, gridlen);

    init_temp<<<gridsize, blocksize>>>(d_past, d_in);

    for(int i=0; i<TOTAL_TIME_ROWS; i++)
    {
        for(int j=0; j<KERNEL_TIME_SKIPS; j++)
        {
            run_sim<<<gridsize, blocksize>>>(d_past, d_in, d_out, d_out_past, d_scratch);               //past and in are just 1 time step apart. similarly d_out and d_out_past are 1 time step apart
            //dpast and din go in to make dout. doutpast is just one step before dout, so we can start our next loop.
            tempo = d_in;
            d_in = d_out;
            d_out = tempo;

            tempo2 = d_past;
            d_past = d_out_past;
            d_out_past = tempo2;
            //cudaMemcpy(d_in, d_out, platedim*sizeof(float), cudaMemcpyDeviceToDevice);
        }

        cudaMemcpy(&h_arr[i*platedim], d_in, platedim*sizeof(float), cudaMemcpyDeviceToHost);           //d_in now actually has the output since we switched the pointers
    }

    FILE *f;
    f = fopen("2d_wave.csv", "w");
    for(int i=0; i<TOTAL_TIME_ROWS; i++)
    {
        for(int j=0;j<platedim; j++)
        {
            fprintf(f, "%.2f,", h_arr[i*platedim+j]);
        }
        fputs("10\n", f);
    }
    fclose(f);


    free(h_arr);
    cudaFree(d_scratch);
    cudaFree(d_out_past);
    cudaFree(d_past);
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
