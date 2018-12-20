##############################################################################
# FILE: Makefile.MPI.c
# DESCRIPTION:
#   Makefile for MPI C Language
# AUTHOR: Dimitris Antoniadis
# LAST REVISED:  4/12/18
# REFERENCES: https://computing.llnl.gov/tutorials/mpi/samples/C/Makefile.MPI.c
###############################################################################

#compiler
CC    =   mpicc

#FLAGS   -Wconversion  used to show implicit conversion warnings
FLAGS   =   -O2 

END_FLAGS = -lm 

all:    mpi_program

clean:  
	/bin/rm -rf	mpi_program
	
mpi_program:	VantagePoint_v_0.4.3.c
	$(CC) $(FLAGS) VantagePoint_v_0.4.3.c -o mpi_program $(END_FLAGS)
