/*
The MIT License (MIT)

Copyright (c) 2014

Athanassios Kintsakis
Contact
athanassios.kintsakis@gmail.com
akintsakis@issel.ee.auth.gr


Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include <float.h>

struct ELEMENT{
	float *coordinates;
};

struct NODE{
	struct ELEMENT Vantage_Point;
	float median;
	struct NODE *Father_Node;
	struct NODE *Left_Node;
	struct NODE *Right_Node;
};

struct SIMPLE_NODE{
	struct ELEMENT Vantage_Point;
	float median;
};

struct knn{
	float *distances;
	struct ELEMENT *Elements;
	float t;
	int k;
	struct ELEMENT query;
	int added;
	int largest_distance_index;
	int *send_array;
};

MPI_Status Stat;
int dimensions;
int MAX_level;

void partition ( float *array, int elements, float pivot, float **arraysmall, float **arraybig, int *endsmall, int *endbig, struct ELEMENT *ElementsArray);
float selection(float *array,int number, struct ELEMENT *ElementsArray);
int correction( float *array, struct ELEMENT *ElementsArray, int length, float pivot);
void create_VP_tree( struct NODE *root, int processId, int noProcesses, struct ELEMENT *ElementsArray, int partLength, int level, MPI_Comm Communicator);
void rearrange( int processId, int noProcesses, int partLength, int FirstBigElementPosition, struct ELEMENT *ElementsArray, MPI_Comm Communicator);
void create_local_VP_tree( struct NODE *root, int length, struct ELEMENT *ElementsArray);
void inform_master( struct SIMPLE_NODE **commonTree, struct NODE *root, int level, int processId);
struct NODE *who_is_my_local_Node( int processId, int noProcesses, struct NODE *root);
void inform_other_nodes( struct SIMPLE_NODE **commonTree, int processId, int level);
void initialize_knn( struct knn *VP_knn, struct ELEMENT Element, int k);
void validate_knn( struct ELEMENT *ElementsArray, int partLength, struct knn *VP_knn);
void update_Node( struct NODE *root, struct SIMPLE_NODE **commonTree, int row_level, int fathers_position);
void search_knn( struct NODE *root, struct knn *VP_knn);
void search_global_knn( struct knn *VP_knn, struct SIMPLE_NODE **commonTree, int level, int position);
void knn( struct NODE *root, struct ELEMENT *ElementsArray, int partLength, struct SIMPLE_NODE **commonTree, int k, struct NODE *local_NODE, struct timeval *fourth, struct timezone *tzp);

/** calculates distance between two elements **/
float dist( float *x, float *y){
	int j;
	float distance = 0;
		for( j = 0; j < dimensions; j++)
			distance = distance + powf( x[j] - y[j], 2);
		distance = sqrtf( distance);
	return distance;
}
/***Kills processes that have no values left in their arrays****/
void removeElement(int *array, int *size, int element)
{
    int i;
    int flag=0;
    for(i=0;i<*size;i++)
    {
        if(flag==1)
            array[i]=array[i+1];
        if(array[i]==element&& flag==0)
        {
            array[i]=array[i+1];
            flag=1;
        }
    }
    *size=*size-1;
}

/****Swaps two values in an array****/
void swap_values(float *array,int x,int y, struct ELEMENT *ElementsArray)
{
    float temp;
	struct ELEMENT tempElement = ElementsArray[x];
    temp=array[x];
    array[x]=array[y];
	ElementsArray[x] = ElementsArray[y];
    array[y]=temp;
	ElementsArray[y] = tempElement;
}

/*****Send random numbers to every node.*****/
void generateElements( struct ELEMENT *ElementsArray, int partLength, int cal)
{
    srand((cal+1)*time(NULL));     	//Generate number to fill the array
    int i, j;
	float a = 100; 					/* Maximum Number that can be generated */
    for(i=0; i<partLength; i++){
		ElementsArray[i].coordinates = ( float*)malloc( dimensions * sizeof(float));
		for( j = 0; j < dimensions; j++){
			ElementsArray[i].coordinates[j] = ((float)rand()/(float)(RAND_MAX)) * a;
		}
	}
}

void calculate_distances( float *distances, struct ELEMENT *ElementsArray, int partLength, float *VP_coordinates){
	int i, j;
	float distance;
	
	for( i = 0; i < partLength; i++){
		distance = 0;
		for( j = 0; j < dimensions; j++)
			distance = distance + powf( ElementsArray[i].coordinates[j] - VP_coordinates[j], 2);
		distance = sqrtf( distance);
		distances[i] = distance;
	}
}
/***Validates the stability of the operation****/
void validation(float median,int partLength,int size, float *numberPart,int processId, MPI_Comm Communicator)
{
    MPI_Bcast(&median,1,MPI_INT,0,Communicator);
	int countMin=0;
    int countMax=0;
    int countEq=0;
    int sumMax,sumMin,sumEq,i;
    for(i=0;i<partLength;i++)
    {
        if(numberPart[i]>median)
            countMax++;
        else if(numberPart[i]<median)
            countMin++;
        else
            countEq++;
    }
    MPI_Reduce(&countMax,&sumMax,1,MPI_INT,MPI_SUM,0,Communicator);
    MPI_Reduce(&countMin,&sumMin,1,MPI_INT,MPI_SUM,0,Communicator);
    MPI_Reduce(&countEq,&sumEq,1,MPI_INT,MPI_SUM,0,Communicator);
    if(processId==0)
    {
        if((sumMax<=size/2)&&(sumMin<=size/2))  //Checks if both the lower and higher values occupy less than 50% of the total array.
            printf("VALIDATION PASSED!\n");
        else
            printf("VALIDATION FAILED!\n");


	printf("Values greater than median: %d\n",sumMax);
        printf("Values equal to median: %d\n",sumEq);
        printf("Values lower than median: %d\n",sumMin);
    }

}

/***Validates the stability of the operation (Single Threaded)****/
void validationST(int median,int size,int *numberPart)
{
	int countMin=0;
    int countMax=0;
    int countEq=0;
    int i;
    for(i=0;i<size;i++)
    {
        if(numberPart[i]>median)
            countMax++;
        else if(numberPart[i]<median)
            countMin++;
        else
            countEq++;
    }
    if((countMax<=size/2)&&(countMin<=size/2))  //Checks if both the lower and higher values occupy less than 50% of the total array.
        printf("VALIDATION PASSED!\n");
    else
        printf("VALIDATION FAILED!\n");

	printf("Values greater than median: %d\n",countMax);
        printf("Values equal to median: %d\n",countEq);
        printf("Values lower than median: %d\n",countMin);
}

void validate_Distance_Position( struct ELEMENT *ElementsArray, float *distances, int partLength, float *coordinates){
	int i,Flag = 0;
	float *new_distances;
	new_distances = (float*)malloc( sizeof(float)*partLength);
	
	calculate_distances( new_distances, ElementsArray, partLength, coordinates);
	
	for( i = 0; i < partLength; i++){
		if( distances[i] != new_distances[i]){
			printf("ELEMENTS REARRAGNMENT INSIDE NODE: FAILED\n");
			Flag = 1;
			break;
		}
	}
	if(Flag != 1)
		printf("ELEMENTS REARRAGNMENT INSIDE NODE: PASSED \n");
}

void validations_Global_rearrangment( int processId , int noProcesses, struct NODE *root, int partLength, struct ELEMENT *ElementsArray){
	float *distances;
	int Flag = 0, i;
	distances = (float*)malloc( partLength * sizeof(float));
	calculate_distances( distances, ElementsArray, partLength, root->Vantage_Point.coordinates);
	if( processId < noProcesses/2){		
		for( i = 0; i < partLength; i++){
			if( distances[i] > root->median){
				Flag = 1;
				break;
			}
		}
	}
	else{		
		for( i = 0; i < partLength; i++){
			if( distances[i] <= root->median){
				Flag = 1;
				break;
			}
		}
	}
	
	if( Flag == 0)
		printf("VALIDATION PASSED.THREAD %d: Elements Rearranged globally correctly\n", processId);
	else
		printf("VALIDATION FAILED distance %f > %f: GLOBAL REARRAGNMENT FAILED \n", distances[i], root->median);
}

/****Part executed only by the Master Node****/
float masterPart( int noProcesses, int processId, int size, int partLength, float *distances, struct ELEMENT *ElementsArray, MPI_Comm Communicator)
{
    int elements,i,keepBigSet,sumSets,finalize,randomNode,k;
    int endSmall=0;
    int dropoutFlag=0;
    int endBig=0;
    int *activeNodes;
    int activeSize=noProcesses;
    int stillActive=1;
    int oldSumSets=-1;
    int checkIdentical=0;
    int useNewPivot=0;
    float *arraySmall,*arrayBig,*arrayToUse, *pivotArray;
	float median, pivot, tempPivot;
	struct ELEMENT *ElementsArrayToUse;
    k=(int)size/2 + 1; //It is done so in order to find the right median in an even numbered array.
    elements=partLength;
    activeNodes=(int *)malloc(noProcesses*sizeof(int));  //we create the array that contains the active nodes.
    arrayToUse=distances;
	ElementsArrayToUse = ElementsArray;
    pivotArray=(float*)malloc(noProcesses*sizeof(float));  //Used for special occasions to gather values different than the pivot.
	ptrdiff_t diff;
    for(i=0;i<activeSize;i++)
    {
        activeNodes[i]=i;
    }
    int randomCounter=0;
    int randomCounter2=0;
    struct timeval first, second, lapsed;
    struct timezone tzp;
    gettimeofday(&first, &tzp);
    for(;;)   //Begin the infinite loop until the median is found.
    {
        int counter=0;
        useNewPivot=0;
        if(stillActive==1&&checkIdentical!=0)  //If i still have values in my array and the Sumed Big Set is identical to the previous one, check for identical values.
        {
            for(i=0;i<elements;i++)
            {
                if(pivot==arrayToUse[i])
                    counter++;
                else
                {
                    useNewPivot=1;
                    tempPivot=arrayToUse[i];
                    break;
                }
            }
        }
        if(checkIdentical!=0)
        {
            int useNewPivotMax=0;
	        MPI_Reduce(&useNewPivot,&useNewPivotMax,1,MPI_INT,MPI_MAX,0,Communicator); //FIRST(OPTIONAL) REDUCE : MAX useNewPivot
            if(useNewPivotMax!=1)    //That means that the only values left are equal to the pivot!
            {
                median=pivot;
                finalize=1;
                MPI_Bcast(&finalize,1,MPI_INT,0,Communicator); //FIRST(OPTIONAL) BROADCAST : WAIT FOR FINALIZE COMMAND OR NOT
                gettimeofday(&second, &tzp);
                if(first.tv_usec>second.tv_usec)
                {
                    second.tv_usec += 1000000;
                    second.tv_sec--;
                }
                lapsed.tv_usec = second.tv_usec - first.tv_usec;
                lapsed.tv_sec = second.tv_sec - first.tv_sec;
                printf("Time elapsed: %lu, %lu s\n", lapsed.tv_sec, lapsed.tv_usec);
                validation(median,partLength,size,distances,processId, Communicator);
                //MPI_Finalize();
                free(pivotArray);
                return median;
            }
            else
            {
                finalize=0;
                int useit=0;
                randomCounter2++;
                MPI_Bcast(&finalize,1,MPI_INT,0,Communicator);
                MPI_Gather(&useNewPivot, 1, MPI_INT, pivotArray, 1, MPI_FLOAT, 0, Communicator); //Gather every value and chose a node to change the pivot.
                for(i=0;i<activeSize;i++)
                {
                    if(pivotArray[i]==1)
                    {
                        if((randomCounter2>1)&&(randomNode!=activeNodes[i]))  //Check if the same node has already been used in a similar operation.
                        {
                            useit=1;
                            randomNode=activeNodes[i];
                            randomCounter2=0;
                            break;
                        }
                        else if(randomCounter2<2)
                        {
                            useit=1;
                            randomNode=activeNodes[i];
                            break;
                        }
                    }
                }
                if(useit!=0)
                    useNewPivot=1;
                else
                    useNewPivot=0;
            }
        }
        if(useNewPivot!=0)
            MPI_Bcast(&randomNode,1,MPI_INT,0,Communicator);  //THIRD(OPTIONAL) BROADCAST : BROADCAST THE SPECIAL NODE
        if(useNewPivot==0)  //if we didnt choose a special Node, choose the node that will pick the pivot in a clockwise manner. Only selects one of the active nodes.
        {
            if(randomCounter>=activeSize)
                randomCounter=0; //Fail safe
            randomNode=activeNodes[randomCounter];
            randomCounter++;			//Increase the counter
            MPI_Bcast(&randomNode,1,MPI_INT,0,Communicator);   //FIRST BROADCAST : SENDING randomnode, who will chose
        }
        if(randomNode==processId)  //If i am to choose the pivot.....
	    {
            if(useNewPivot==0)
            {
                srand(time(NULL));
                pivot=arrayToUse[rand() % elements];
                MPI_Bcast(&pivot,1,MPI_FLOAT,0,Communicator); //SECOND BROADCAST : SENDING PIVOT   k ton stelnw sto lao
	        }
            else
            {
                MPI_Bcast(&tempPivot,1,MPI_FLOAT,0,Communicator); //SECOND BROADCAST : SENDING PIVOT   k ton stelnw sto lao
                pivot=tempPivot;
            }
        }
        else //If not.. wait for the pivot to be received.
            MPI_Bcast(&pivot,1,MPI_FLOAT,randomNode,Communicator);  // SECOND BROADCAST : RECEIVING PIVOT
        if(stillActive==1)  //If i still have values in my array.. proceed
        {
            partition(arrayToUse,elements,pivot,&arraySmall,&arrayBig,&endSmall,&endBig, ElementsArrayToUse);  //I partition my array  // endsmall=number of elements in small array, it may be 0            
        }
        else  //If i'm not active endBig/endSmall has zero value.
        {
            endBig=0;
            endSmall=0;
        }
        sumSets=0;
	    //We add the bigSet Values to decide if we keep the small or the big array
	    MPI_Reduce(&endBig,&sumSets,1,MPI_INT,MPI_SUM,0,Communicator);  //FIRST REDUCE : SUM OF BIG
        MPI_Bcast(&sumSets,1,MPI_INT,0,Communicator);
        if(oldSumSets==sumSets)
            checkIdentical=1;
        else
        {
            oldSumSets=sumSets;
            checkIdentical=0;
        }
	    //hmetabliti keepBigSet 0 h 1 einai boolean k me autin enimerwnw ton lao ti na kratisei to bigset h to smallset
	    if(sumSets>k)   //an to sumofbigsets > k tote krataw to big SET
	    {
            keepBigSet=1; //to dilwnw auto gt meta tha to steilw se olous
            if(endBig==0)
                dropoutFlag=1; //wraia.. edw an dw oti to bigset mou einai 0.. alla prepei na kratisw to bigset sikwnw auti ti simaia pou simainei tha ginw inactive ligo pio katw tha to deis
            else
            {
                arrayToUse=arrayBig; //thetw ton neo pinaka na einai o big
				ElementsArrayToUse = &ElementsArrayToUse[endSmall];
                elements=endBig; //thetw arithmo stoixeiwn iso me tou big
            }
	    }
	    else if(sumSets<k) //antistoixa an to sumofbigsets < k tote krataw to small set
	    {
		    keepBigSet=0;
		    k=k-sumSets;
		    if(endSmall==0)
                dropoutFlag=1; //antistoixa koitaw an tha ginw inactive..
		    else
		    {
		    	arrayToUse=arraySmall; //dinw times..
				ElementsArrayToUse = ElementsArrayToUse;
		    	elements=endSmall;
		    }
	    }
	    else  //edw simainei k=sumofbigsetes ara briskw pivot k telos
	    {
		    median=pivot;
		    finalize=1; //dilwnw finalaize =1
		    MPI_Bcast(&finalize,1,MPI_INT,0,Communicator); //to stelnw se olous, oi opoioi an laboun finalize =1 tote kaloun MPI finalize k telos
		    gettimeofday(&second, &tzp);
            if(first.tv_usec>second.tv_usec)
            {
                second.tv_usec += 1000000;
                second.tv_sec--;
            }
            lapsed.tv_usec = second.tv_usec - first.tv_usec;
            lapsed.tv_sec = second.tv_sec - first.tv_sec;
            printf("Time elapsed: %lu, %lu s\n", lapsed.tv_sec, lapsed.tv_usec);
		    validation(median,partLength,size,distances,processId, Communicator);
            //MPI_Finalize();
            free(pivotArray);
            return median;
        }
        finalize=0; //an den exw mpei sta if den exw steilei timi gia finalize.. oi alloi omws perimenoun na laboun kati, stelnw loipon to 0 pou simainei sunexizoume
        MPI_Bcast(&finalize,1,MPI_INT,0,Communicator);	//SECOND BROADCAST : WAIT FOR FINALIZE COMMAND OR NOT
        //edw tous stelnw to keepbigset gia na doun ti tha dialeksoun
	    MPI_Bcast(&keepBigSet,1,MPI_INT,0,Communicator);    //THIRD BROADCAST: SEND keepBigset boolean
        if(dropoutFlag==1 && stillActive==1) //edw sumfwna me to dropoutflag pou orisame prin an einai 1 kalw tin sinartisi pou me petaei apo ton pinaka. episis koitaw na eimai active gt an me exei idi petaksei se proigoumeni epanalispi tote den xreiazetai na me ksanapetaksei
        {
            stillActive=0;
            removeElement(activeNodes, &activeSize, 0);
        }
        int flag;
        //edw perimenw na akousw apo ton kathena an sunexizei active h oxi.. an oxi ton petaw.. an einai idi inactive apo prin stelnei kati allo (oxi 1)k den ton ksanapetaw
        for(i=0;i<activeSize;i++)
        {
            if(activeNodes[i]!=0)
            {
                MPI_Recv(&flag,1,MPI_INT,activeNodes[i],1,Communicator,&Stat);  //FIRST RECEIVE : RECEIVE active or not
                if(flag==1)
                    removeElement(activeNodes, &activeSize, activeNodes[i]);
            }
        }
    }
}

/***Executed only by Slave nodes!!*****/
void slavePart( int processId, int partLength, float *distances, int size, struct ELEMENT *ElementsArray, MPI_Comm Communicator)
{
	int dropoutflag,elements,i,sumSets,finalize,keepBigSet,randomNode;
    int endSmall=0;
    int endBig=0;
    float *arraySmall,*arrayBig,*arrayToUse;
	arrayToUse=distances;
	elements=partLength;
	int stillActive=1;
	float *pivotArray;
    int oldSumSets=-1;
    int checkIdentical=0;
    int useNewPivot;
	float pivot, tempPivot;
	struct ELEMENT *ElementsArrayToUse;
	ElementsArrayToUse = ElementsArray;
	ptrdiff_t diff;
	
	for(;;)
	{
        finalize=0;
        int counter=0;
        useNewPivot=0;
        if(stillActive==1&&checkIdentical!=0)  //If i still have values in my array..   If the Sumed Big Set is identical to the previous one, check for identical values.
        {
            for(i=0;i<elements;i++)
            {
                if(pivot==arrayToUse[i])
                    counter++;
                else
                {
                    useNewPivot=1;
                    tempPivot=arrayToUse[i];
                    break;
                }
            }
        }
        if(checkIdentical!=0)
        {
            int useNewPivotMax=0;
            MPI_Reduce(&useNewPivot,&useNewPivotMax,1,MPI_INT,MPI_MAX,0,Communicator);
            MPI_Bcast(&finalize,1,MPI_INT,0,Communicator);//an o master apo to keepbigset k apo to count apofasisei oti teleiwsame mou stelnei 1, alliws 0 sunexizoume
            if(finalize==1)
            {
                int median=0;
                validation(median,partLength,size,distances,processId, Communicator);
                //MPI_Finalize();
                return ;
            }
            else
            {
                MPI_Gather(&useNewPivot, 1, MPI_INT, pivotArray, 1, MPI_FLOAT, 0, Communicator);
            }
        }
        MPI_Bcast(&randomNode,1,MPI_INT,0,Communicator); //FIRST BROAD CAST : RECEIVING RANDOM NODE, perimenw na dw poios einaito done
        if(randomNode!=processId) //means I am not the one to chose pivot.. so I wait to receive the pivot
            MPI_Bcast(&pivot,1,MPI_FLOAT,randomNode,Communicator);	//SECOND BROADCAST : RECEIVING PIVOT
        else if(randomNode==processId) //I am choosing suckers
        {
            if(useNewPivot==0)
            {
                srand(time(NULL));
                pivot=arrayToUse[rand() % elements];
                MPI_Bcast(&pivot,1,MPI_FLOAT,processId,Communicator); //SECOND BROADCAST : SENDING PIVOT   k ton stelnw sto lao
            }
            else
            {
                MPI_Bcast(&tempPivot,1,MPI_FLOAT,processId,Communicator); //SECOND BROADCAST : SENDING PIVOT   k ton stelnw sto lao
                pivot=tempPivot;
            }
        }
        if(stillActive==1)   //an eksakolouthw na eimai active, trexw tin partition.. k to count kommati to opio eimape kapou exei problima
        {
            partition(arrayToUse,elements,pivot,&arraySmall,&arrayBig,&endSmall,&endBig, ElementsArrayToUse);			
        }
        else
        {
            endBig=0;
            endSmall=0;
        }
        //an eimai inactive stelnw endbig=0 gia to bigset pou den epireazei
        sumSets=0;
        MPI_Reduce(&endBig,&sumSets,1,MPI_INT,MPI_SUM,0,Communicator); //FIRST REDUCE : SUM OF BIG, stelnw ola ta bigset gia na athroistoun sotn master
        MPI_Bcast(&sumSets,1,MPI_INT,0,Communicator);
        if(oldSumSets==sumSets)
            checkIdentical=1;
        else
        {
            oldSumSets=sumSets;
            checkIdentical=0;
        }
        MPI_Bcast(&finalize,1,MPI_INT,0,Communicator);//an o master apo to keepbigset k apo to count apofasisei oti teleiwsame mou stelnei 1, alliws 0 sunexizoume
        if(finalize==1)
        {
            int median=0;
            validation(median,partLength,size,distances,processId, Communicator);
            //MPI_Finalize();
            return ;
        }
        MPI_Bcast(&keepBigSet,1,MPI_INT,0,Communicator);//THIRD BROADCAST: Receive keepBigset boolean, edw lambanw an krataw to mikro i megalo set.
            //afou elaba ton keepbigset an eimai active krataw enan apo tous duo pinake small h big.. alliws den kanw tpt
            //edw antistoixa allazw tous pointers, k eksetazw an exw meinei xwris stoixeia tin opoia periptwsi sikwnw to dropoutflag k pio katw tha dilwsw na ginw inactive
        if(stillActive==1)
        {
            if(keepBigSet==1)
            {
                if(endBig==0)
                    dropoutflag=1;
                else
                {
                    arrayToUse=arrayBig;
					ElementsArrayToUse = &ElementsArrayToUse[endSmall];
                    elements=endBig;
                }
            }
            else if(keepBigSet==0)
            {
                if(endSmall==0)
                    dropoutflag=1;
                else
                {
                    arrayToUse=arraySmall;
					ElementsArrayToUse = ElementsArrayToUse;
                    elements=endSmall;
                }
            }
        }
        //edw einai ligo periploka grammeno, isws exei perita mesa alla, an eimai active k thelw na ginw inactive einai i prwti periptwsi, h deuteri einai eimai inactive hdh k i triti einai sunexizw dunamika
        if(dropoutflag==1 && stillActive==1)
        {
            MPI_Send(&dropoutflag,1,MPI_INT,0,1,Communicator); //FIRST SEND : send active or not;
            stillActive=0;
        }
        else if(stillActive==0)
        {
            dropoutflag=-1;
            MPI_Send(&dropoutflag,1,MPI_INT,0,1,Communicator); //FIRST SEND : send active or not;
        }
        else
        {
            dropoutflag=0;
            MPI_Send(&dropoutflag,1,MPI_INT,0,1,Communicator); //FIRST SEND : send active or not;
        }
    }
}


/*****MAIN!!!!!!!!!!*****/
int main (int argc, char **argv){
    int processId,noProcesses,size,partLength, power, i, j, k, q;
    struct ELEMENT *ElementsArray;
	struct timeval first, second, third, fourth, lapsed;
	struct timezone tzp;

    size=atoi(argv[1]);
    MPI_Init (&argc, &argv);	/* starts MPI */
    MPI_Comm_rank (MPI_COMM_WORLD, &processId);	/* get current process id */
    MPI_Comm_size (MPI_COMM_WORLD, &noProcesses);	/* get number of processes */
    
	
	dimensions = atoi(argv[2]);
	/* CONSTRAINTS */
	if(processId==0){
		/* Execution Constraint */
		if( argc != 4){
			printf("ERROR:Execution requires exact 1 argument\n");
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
		/* Number Of Processes Constraint */
		/* if( noProcesses%2 != 0 || noProcesses == 1){
			printf("ERROR: Number Of Processes must be multiple of 2\n");
			MPI_Abort(MPI_COMM_WORLD, 2);
		} */
		/* Dimensions Constraints */
		if(  dimensions < 1){
			printf("ERROR: Dimensions non-positive\n");
			MPI_Abort(MPI_COMM_WORLD, 2);
		}
	}
	/* END OF CONSTRAINTS */
	
	power = atoi(argv[1]);	
	MAX_level = logf(noProcesses)/logf(2); 
	size = 1 << power;							/* size of Elements 2^power */
	partLength = size/noProcesses; 				/* Length of Array of Elements in each Node */
	ElementsArray =  ( struct ELEMENT*)malloc( partLength * sizeof( struct ELEMENT));	/* Allocate space for partLength Elements */	
	
	if(processId==0)
    {
        printf("size: %d processes: %d\n",size,noProcesses);
        if(noProcesses>1)
        {
            generateElements( ElementsArray, partLength, processId);	/* Generate Elements */
        }
        else
        {	
            ElementsArray =  ( struct ELEMENT*)malloc( partLength * sizeof( struct ELEMENT));	/* Allocate space for partLength Elements */	
            generateElements( ElementsArray, partLength, processId);
            struct timeval first, second, lapsed;
            struct timezone tzp;
            gettimeofday(&first, &tzp);
			struct NODE root;
			root.Father_Node = NULL;
            create_local_VP_tree( &root, partLength, ElementsArray);
            gettimeofday(&second, &tzp);
            if(first.tv_usec>second.tv_usec)
            {
                second.tv_usec += 1000000;
                second.tv_sec--;
            }
            lapsed.tv_usec = second.tv_usec - first.tv_usec;
            lapsed.tv_sec = second.tv_sec - first.tv_sec;
            //validationST(median,size,distances);
            //printf("Time elapsed: %lu, %lu s\n", lapsed.tv_sec, lapsed.tv_usec);
           // printf("Median: %d\n",median);
            //free(distances);
           // MPI_Finalize();
            //return 0;
			k = 1<<atoi(argv[3]);
			MPI_Barrier(MPI_COMM_WORLD);
			if(processId == 0){
				gettimeofday(&third, &tzp);
			}
			struct knn *VP_knn = (struct knn*)malloc(partLength * sizeof(struct knn));
			//knn( &root, ElementsArray, partLength, commonTree, k + 1, local_NODE, &fourth, &tzp);
			for( i = 0; i < partLength; i++){
						initialize_knn( &VP_knn[i], ElementsArray[i],  k+1);
						search_knn( &root, &VP_knn[i]);
						//search_global_knn( &VP_knn[i], commonTree, 0, 0);	
			}
			
			MPI_Barrier(MPI_COMM_WORLD);
			if(processId == 0){
				gettimeofday(&fourth, &tzp);
			}
			if(processId == 0){
				sleep(2);
				if(first.tv_usec>second.tv_usec)
				{
					second.tv_usec += 1000000;
					second.tv_sec--;
				}
				lapsed.tv_usec = second.tv_usec - first.tv_usec;
				lapsed.tv_sec = second.tv_sec - first.tv_sec;
				printf("Time elapsed of Tree Creation: %lu, %lu s\n", lapsed.tv_sec, lapsed.tv_usec);
				if(third.tv_usec>fourth.tv_usec)
				{
					second.tv_usec += 1000000;
					second.tv_sec--;
				}
				lapsed.tv_usec = fourth.tv_usec -third.tv_usec;
				lapsed.tv_sec = fourth.tv_sec - third.tv_sec;
				printf("Time elapsed of All-knn search: %lu, %lu s\n", lapsed.tv_sec, lapsed.tv_usec);
			}
			MPI_Finalize();	
			free( ElementsArray);
			return 0;
			
        }
    }
    else
    {
        generateElements( ElementsArray, partLength, processId);	/* Generate Elements */
    }
	
	/* print arrays of all Nodes */
	/* for( int i = 0; i < noProcesses; i++){
		if( processId == i){
			printf("THREAD %d: \n", processId);
			for( int j = 0; j < partLength; j++){
				printf("[");
				for( int k = 0; k < dimensions; k++)
					printf("%.2f ", ElementsArray[j].coordinates[k]);
				printf("]");
			}
			printf("\n");
		}
		MPI_Barrier(MPI_COMM_WORLD);
	} */
	
	MPI_Barrier(MPI_COMM_WORLD);
	if(processId == 0){
		gettimeofday(&first, &tzp);
	}
	struct NODE root;
	root.Father_Node = NULL;	
	
	if( processId == 0){
		create_VP_tree( &root, processId, noProcesses, ElementsArray, partLength, MAX_level, MPI_COMM_WORLD);
	}
	else{
		create_VP_tree( &root, processId, noProcesses, ElementsArray, partLength, MAX_level, MPI_COMM_WORLD);
	}
	//MPI_Barrier(MPI_COMM_WORLD);
	//sleep(1);
	
	/* Allocate a 2D array which represents common tree */
	/* array[0][1] 								root							*
	*  array[1][2]    			Left Node					Right Node		   	*
	*  array[2][4]  	Left Node      Right Node   Left Node      Right Node 	*
	*  array[3][8]         /\			   /\           /\              /\      */

	struct SIMPLE_NODE **commonTree;
	commonTree = (struct SIMPLE_NODE**)malloc((MAX_level ) * sizeof( struct SIMPLE_NODE*));
	for( i = 0; i < MAX_level + 1; i++){
		commonTree[i] = (struct SIMPLE_NODE*)malloc(pow(2, i) * sizeof(struct SIMPLE_NODE));
	}	
	
	inform_master( commonTree, &root, MAX_level, processId);
	//MPI_Barrier(MPI_COMM_WORLD);
	//sleep(1);
	
	struct NODE *local_NODE = who_is_my_local_Node( processId, noProcesses, &root);
	
	
	//MPI_Barrier(MPI_COMM_WORLD);
	//sleep(1);
	/* for( i = 0; i < noProcesses; i++){
		if( processId == i){
			printf("THREAD %d: VANTAGE POINT is", processId);
			for( k = 0; k < dimensions; k++)
				printf(" %f", local_NODE->Vantage_Point.coordinates[k]);
			printf(" Median is %f\n", local_NODE->median);
		}
		MPI_Barrier(MPI_COMM_WORLD);		
	}  */
	inform_other_nodes( commonTree, processId, MAX_level);	
	//MPI_Barrier(MPI_COMM_WORLD);
	//sleep(1);
	/* for( q = 0; q < noProcesses; q++){
		if( processId == q){
			for( i = 0; i < MAX_level; i++){
				for( j = 0; j < pow(2, i); j++){
				printf("[%d][%d] Vantage Point :", i, j);
					for( k = 0; k < dimensions; k++)
						printf(" %f", commonTree[i][j].Vantage_Point.coordinates[k]);
					printf(", Median is %f\n",  commonTree[i][j].median);
				}
			}
		}
		MPI_Barrier( MPI_COMM_WORLD);
	} */
	MPI_Barrier(MPI_COMM_WORLD);	
	if(processId == 0){
		gettimeofday(&second, &tzp);
	}
	
	
	k = 1<<atoi(argv[3]);
	MPI_Barrier(MPI_COMM_WORLD);
	if(processId == 0){
		gettimeofday(&third, &tzp);
	}
	struct knn *VP_knn = (struct knn*)malloc(partLength * sizeof(struct knn));
	//knn( &root, ElementsArray, partLength, commonTree, k + 1, local_NODE, &fourth, &tzp);
	for( i = 0; i < partLength; i++){
				initialize_knn( &VP_knn[i], ElementsArray[i],  k+1);
				search_knn( &root, &VP_knn[i]);
				//search_global_knn( &VP_knn[i], commonTree, 0, 0);	
	}
	
	MPI_Barrier(MPI_COMM_WORLD);
	if(processId == 0){
		gettimeofday(&fourth, &tzp);
	}
	if(processId == 0){
		sleep(2);
		if(first.tv_usec>second.tv_usec)
		{
			second.tv_usec += 1000000;
			second.tv_sec--;
		}
		lapsed.tv_usec = second.tv_usec - first.tv_usec;
		lapsed.tv_sec = second.tv_sec - first.tv_sec;
		printf("Time elapsed of Tree Creation: %lu, %lu s\n", lapsed.tv_sec, lapsed.tv_usec);
		if(third.tv_usec>fourth.tv_usec)
		{
			second.tv_usec += 1000000;
			second.tv_sec--;
		}
		lapsed.tv_usec = fourth.tv_usec -third.tv_usec;
		lapsed.tv_sec = fourth.tv_sec - third.tv_sec;
		printf("Time elapsed of All-knn search: %lu, %lu s\n", lapsed.tv_sec, lapsed.tv_usec);
	}
	MPI_Finalize();	
    free( ElementsArray);
    return 0;
}

/*---------CREATE GLOBAL-LOCAL VANTAGE POINT TREE  FUNCTIONS-----------*/
/*** Create Vantage Point tree ***/
void create_VP_tree( struct NODE *root, int processId, int noProcesses, struct ELEMENT *ElementsArray, int partLength, int level, MPI_Comm Communicator){
	int VP_index, size, i, j;
	int FirstBigElementPosition;
	float temp, median;
	float *distances;
	
	root->Left_Node = NULL;
	root->Right_Node = NULL;
	if(level > 0) size = pow(2, level)*partLength;
	
	//printf("THREAD %d: creating Vantage Point tree : LEVEL %d \n", processId, level); 
	root->Vantage_Point.coordinates = ( float*)malloc( dimensions * sizeof(float)); 	/* Allocate Memory for Coordinates */
		
	if( level > 0){
		if( processId == 0){ 					/* If i am the Master */
			srand( time(NULL));					/* I select Vantage Point  randomly*/
			VP_index = rand() % partLength;		/* Vantage Point index */
			for( i = 0; i < dimensions; i++)	/* Store Vantage Point Coordinates */
				root->Vantage_Point.coordinates[i] = ElementsArray[VP_index].coordinates[i];
			
			printf("THREAD %d: Vantage Point is:", processId);		/*Print Vantage Point */
			for( i = 0; i < dimensions; i++){
				//printf(" %.2f", root->Vantage_Point.coordinates[i]);
				//if( i == dimensions - 1) printf("\n");
			}
			
			for( i = 0; i < dimensions; i++){	/* BROADCAST Vantage Point: ANNOUNCE */
				temp = root->Vantage_Point.coordinates[i];
				MPI_Bcast( &temp, 1, MPI_FLOAT, 0, Communicator);
			}
			
		}
		else{									/* if I am not the master */
			//sleep(1);							//Dimitris: This sleep is used to synchronize printfs ***REMOVE IT***				
			//printf("THREAD %d: Waiting Vantage Point to be announced\n", processId);
			for( i = 0; i < dimensions; i++){	/* BROADCAST Vantage Point: RECEIVE */
				MPI_Bcast( &temp, 1, MPI_FLOAT, 0, Communicator);
				root->Vantage_Point.coordinates[i] = temp;			
			}
			//printf("THREAD %d: Vantage Point is:", processId);		/*Print Vantage Point */
			for( i = 0; i < dimensions; i++){
				//printf(" %.2f", root->Vantage_Point.coordinates[i]);
				//if( i == dimensions - 1) printf("\n");
			}
		}
		
			
		
		distances = ( float*)malloc( partLength * sizeof(float));
		calculate_distances( distances, ElementsArray, partLength, &(root->Vantage_Point.coordinates[0]));
		
		/* print arrays of all Nodes */
		/* for( int i = 0; i < noProcesses; i++){
			if( processId == i){
				printf("THREAD %d: \n", processId);
				for( int j = 0; j < partLength; j++){
					printf("%.2f ", distances[j]);
				}
				printf("\n");
			}
			MPI_Barrier( Communicator);
			sleep(1);
		}   */
		
		if( processId == 0){
			median = masterPart( noProcesses, processId, size, partLength, distances, ElementsArray, Communicator);
			//printf("Median: %f\n", median);
		}
		else{
			slavePart( processId, partLength, distances, size, ElementsArray, Communicator);
		}
		
		MPI_Bcast( &median, 1, MPI_FLOAT, 0, Communicator);		/* Inform All the THREADS about median */
		//MPI_Barrier(Communicator);
		root->median = median;
		
		FirstBigElementPosition = correction( distances,ElementsArray, partLength, median);
		
		if(FirstBigElementPosition == partLength)
			printf("THREAD: %d: NO BIG ELEMENTS\n", processId);
		else
			printf("THREAD: %d FirstBigElementPosition: %d is with value %f \n",processId, FirstBigElementPosition,distances[FirstBigElementPosition]);
		
		//printf("THREAD %d: FBE: %d\n", processId, FirstBigElementPosition);
		/* print arrays of all Nodes */
		 /* for( int i = 0; i < noProcesses; i++){
			if( processId == i){
				printf("THREAD %d: \n", processId);
				for( int j = 0; j < partLength; j++){
					printf("%.2f ", distances[j]);
				}
				printf("\n");
			}
			MPI_Barrier( Communicator);
		}  */
		//MPI_Barrier(Communicator);
		
		validate_Distance_Position( ElementsArray, distances, partLength,&(root->Vantage_Point.coordinates[0]));	/* This Function Validates that Elements have been rearranged according to distances */
		//MPI_Barrier(Communicator);
		//sleep(1);
		rearrange( processId, noProcesses, partLength, FirstBigElementPosition, ElementsArray, Communicator);
		//MPI_Barrier(Communicator);
		//sleep(1);
		//printf("Before Val \n");
		validations_Global_rearrangment( processId , noProcesses, root, partLength, ElementsArray);
		free(distances);
		//MPI_Barrier(Communicator);
		//sleep(1);
		
		
		
		
		/* Create New Communicator */
		/* Split the old Communicator in two teams */
		int color = processId/(noProcesses/2);
		MPI_Comm newCommunicator;
		MPI_Comm_split(Communicator, color, processId, &newCommunicator);
		int newProcessId;
		int newSize;
		MPI_Comm_rank( newCommunicator, &newProcessId);
		MPI_Comm_size( newCommunicator, &newSize);
		
		
		

		
		
		//MPI_Barrier(MPI_COMM_WORLD);
		//sleep(1);
		if( processId < noProcesses/2){
			root->Left_Node = (struct NODE*)malloc(1*sizeof(struct NODE));
			//root->Left_Node->Father_Node = (struct NODE*)malloc(1* sizeof(struct NODE));
			root->Left_Node->Father_Node = root;
			create_VP_tree( root->Left_Node, newProcessId, newSize, ElementsArray, partLength, level - 1, newCommunicator);
		}
		else{ 
			root->Right_Node = (struct NODE*)malloc(1*sizeof(struct NODE));
			//root->Right_Node->Father_Node = (struct NODE*)malloc(1* sizeof(struct NODE));
			root->Right_Node->Father_Node = root;
			create_VP_tree( root->Right_Node, newProcessId, newSize, ElementsArray, partLength, level - 1, newCommunicator);
		}
		
	}
	else{
		create_local_VP_tree( root, partLength, ElementsArray);
	}
}		

/*** Create local Vantage Point Tree ***/
void create_local_VP_tree( struct NODE *root, int length, struct ELEMENT *ElementsArray){
	float *distances;
	int i;
	root->Vantage_Point.coordinates = ( float*)malloc( dimensions * sizeof(float)); 	/* Allocate Memory for Coordinates */
	
	if(length > 1){
		int VP_index;
		srand( time(NULL));					/* I select Vantage Point  randomly*/
		VP_index = rand() % length;		/* Vantage Point index */
		for( i = 0; i < dimensions; i++)	/* Store Vantage Point Coordinates */
			root->Vantage_Point.coordinates[i] = ElementsArray[VP_index].coordinates[i];
			
		distances = ( float*)malloc(sizeof(float)*length);	
		calculate_distances( distances, ElementsArray, length, &(root->Vantage_Point.coordinates[0]));
		root->median = selection( distances, length, ElementsArray);
		
		root->Left_Node = (struct NODE*)malloc(1*sizeof(struct NODE));
		root->Right_Node = (struct NODE*)malloc(1*sizeof(struct NODE));
		root->Left_Node->Father_Node = (struct NODE*)malloc(1* sizeof(struct NODE));
		root->Right_Node->Father_Node = (struct NODE*)malloc(1* sizeof(struct NODE));
		root->Left_Node->Father_Node = root;
		root->Right_Node->Father_Node = root;
		/** TODO: openMP **/
		create_local_VP_tree( root->Left_Node, length/2,  ElementsArray);
		create_local_VP_tree( root->Right_Node, length/2, &ElementsArray[length/2]);
	}
	else{
		
		
		root->median = -1;					/*This is a leaf */
		for( i = 0; i < dimensions; i++)	/* Store Vantage Point Coordinates */
			root->Vantage_Point.coordinates[i] = ElementsArray[0].coordinates[i];
		root->Left_Node = NULL;
		root->Right_Node = NULL;
	}	
}

/*** Rearrange Elements lower/bigger than median ***/
void rearrange( int processId, int noProcesses, int partLength, int FirstBigElementPosition, struct ELEMENT *ElementsArray, MPI_Comm Communicator){
	
	/* FirstBigElementPosition shows the position of the first bigger or equal element
	*  than the median. Indexing begins from 0. So the length of elemnts lower than 
	*  median in the array is equal to the variable FirstBigElementPosition
	*/
	
	/*Variables Declaration */
	int sendLength, receiveLength, exchangeLength, difference, dest, source, counter = 0, i, j;
	
	//MPI_Status Stat;   // required variable for receive routines
	float *buffer;
	int myId = processId;
	
	
	
	/* FIRST: NODE i communicates with NODE i + (noProcesses in the group/2)
	*  to find out the length of the arrays that they will interchange */
	if( myId < (noProcesses)/2){
		
		int	sendIndex, bufferIndex = 0;
		
		/* Initialization of Variables */
		sendIndex = FirstBigElementPosition;
		sendLength = partLength - FirstBigElementPosition;
		
		
		/* buffer will never be bigger than sendLength */
		if( sendLength > 0 ) //check if there is no need to allocate new buffer
			buffer = (float*)malloc(sendLength*sizeof(float));
		
		/* Each Node waits until the previous node informs him about the dest to swap with elements */
		if( myId > 0){ 		/* Master Thread has always processId equal to 0 */
			int previous_node = myId - 1;
			printf("THREAD %d: Waiting thread %d\n", myId, previous_node);
			MPI_Recv(&dest, 1, MPI_INT, previous_node, 6, Communicator, &Stat);
			printf("THREAD %d: dest received from %d destination is %d\n", myId, previous_node, dest);
			source = dest;
			if( (dest > noProcesses - 1) || dest == -1){
				/* exhange has been completed, sendLength = 0 */
				sendLength = -1;
				dest = -1;
			}
		}
		else{ /* unless i am the Master who initiates this activity */
			//printf("THREAD: %d, dest: %d\n", myId, (noProcesses)/2);
			dest = (noProcesses)/2;
			source = dest;
		}
		
		
		/* If sendLength == 0 Inform the other to continue with the next one */
		if( sendLength == 0){
			/* Send the Length of available numbers i can send */
			MPI_Send(&sendLength, 1, MPI_INT, dest, 1, Communicator);
			//printf("THREAD %d: Iteration %d:  Length sent to dest %d: is %d: 859 \n", myId, counter, dest, sendLength);
			
			/* Receive the available numbers the destination can send */
			MPI_Recv(&receiveLength, 1, MPI_INT, source,2, Communicator, &Stat);
			//printf("THREAD %d: Iteration %d:  Length received from dest %d: is %d: 864 \n", myId, counter, dest, receiveLength);
		}
		
		counter = 0;
		//printf("THREAD %d: sendLength %d\n", myId, sendLength);
		/* sendLength > 0 */
		while( sendLength > 0){
			
			counter++;
			source = dest;
			/* Send the Length of available numbers i can send */
			MPI_Send(&sendLength, 1, MPI_INT, dest, 1, Communicator);
			//printf("THREAD %d: Iteration %d:  Length sent to dest %d: is %d: 873 \n", myId, counter, dest, sendLength);
			
			/* Receive the available numbers the destination can send */
			MPI_Recv(&receiveLength, 1, MPI_INT, source, 2, Communicator, &Stat);
			//printf("THREAD %d: Iteration %d:  Length received from dest %d: is %d: 877 \n", myId, counter, dest, receiveLength);
			
			/* Calculate the difference between sendLength and receiveLength */
			difference = sendLength - receiveLength;
			
			if( difference >= 0)
				exchangeLength = receiveLength;
			else
				exchangeLength = sendLength;
		
			
			printf(" I am %d I send data to %d SEND LENGTH: %d RECEIVE LENGTH: %d EXCHANGE LENGTH: %d\n", myId, dest, sendLength, receiveLength, exchangeLength);
			
			if( exchangeLength > 0){
				
				for( j = 0; j < dimensions; j++){
					for( i = sendIndex; i < partLength; i++)
						buffer[bufferIndex + i - sendIndex] = ElementsArray[i].coordinates[j];
					
					/* send exchanged data */			
					MPI_Send( &(buffer[bufferIndex]), exchangeLength, MPI_FLOAT, dest, 3, Communicator);
					//printf("THREAD %d: Iteration %d:  Exchange sent to dest %d: is %d: 893 \n", myId, counter, dest, exchangeLength);
					
					/* receive exchanged data */
					MPI_Recv( &(buffer[bufferIndex]), exchangeLength, MPI_FLOAT, source, 4, Communicator, &Stat);
					//printf("THREAD %d: Iteration %d:  Exchange received from dest %d: is %d: 897 \n", myId, counter, dest, exchangeLength);
					
					/* store the exchanged data */
					for( i = sendIndex; i < partLength; i++)
						ElementsArray[i].coordinates[j] = buffer[bufferIndex + i - sendIndex]; 
				}
				
				/* Increase sendIndex */
				sendIndex = sendIndex + exchangeLength;
				
				/* Increase bufferIndex */
				bufferIndex = bufferIndex + exchangeLength;
				
				if( difference >= 0){ 
					/* If difference is >= 0 then my partner is fine so i will communicate with the next */					
					dest++;
				}
				
				/* update sendLength */
				sendLength = sendLength - exchangeLength;
			}
			else /* in this case the other has no elements to send */
				dest++;
				
						
		}
		
		//print( _NODE->array, _NODE->arrayLength);
		
		/* In case no rearrangement happened */
		if( counter == 0)
			printf("THREAD %d: No rearrangement needed\n", myId);
		
		/* If i am not the P/2 - 1 Node inform the next node to start communication */
		if( myId < noProcesses/2 - 1){
			int next_node = myId + 1;
			printf("THREAD %d: Sending dest to %d\n", myId, next_node);
			MPI_Send(&dest, 1, MPI_INT, next_node, 6, Communicator);
			printf("THREAD %d: dest sent to %d\n", myId, next_node);
		}
		free(buffer);
			
		
	}
	else if( myId >= (noProcesses)/2){
		
		counter = 0;
		int	receiveIndex;
		float temp;
		
		sendLength = FirstBigElementPosition;
		receiveIndex = 0; // in this case receiveIndex is also bufferIndex
		
		/* buffer will never be bigger than sendLength */
		if( sendLength > 0 ) //check if there is no need to allocate new buffer
			buffer = (float*)malloc(sendLength*sizeof(float));
		
		/* I wait until previous node tells me to start */
		if( myId != (noProcesses)/2){
			int previous_node = myId - 1;
			printf("THREAD %d: Waiting thread %d\n", myId, previous_node);
			MPI_Recv(&dest, 1, MPI_INT,previous_node, 6, Communicator, &Stat);
			printf("THREAD %d: dest received from %d dest is %d\n", myId, previous_node, dest);
			source = dest;
			if( (dest > noProcesses/2 - 1) || dest == -1){
				/* exhange has been completed, sendLength = 0 */
				sendLength = -1;
				dest = -1;
			}
		}
		else{ /* unless i am node : P/2 */
			//printf("THREAD: %d, dest: %d\n", myId, myId - (noProcesses)/2);
			dest = 0;
			source = dest;
		}
		
		/* If sendLength == 0 Inform the other to continue with the next one */
		if( sendLength == 0){			
			/* Receive the available numbers the destination can send */
			MPI_Recv(&receiveLength, 1, MPI_INT, source, 1, Communicator, &Stat);
			//printf("THREAD %d: Iteration %d:  Length sent to dest %d: is %d: 965 \n", myId, counter, dest, sendLength);
			
			/* Send the Length of available numbers i can send */
			MPI_Send(&sendLength, 1, MPI_INT, dest, 2, Communicator);
			//printf("THREAD %d: Iteration %d:  Length received from dest %d: is %d: 969 \n", myId, counter, dest, receiveLength);
			
			
		}
		
		//printf("THREAD %d: sendLength %d\n", myId, sendLength);
		while( sendLength > 0){
			counter++;
			source = dest;
			
			/* Receive the Length of available numbers dest wants to send */
			MPI_Recv(&receiveLength, 1, MPI_INT, source, 1, Communicator, &Stat);
			//printf("THREAD %d: Iteration %d:  Length received from dest %d: is %d: 979 \n", myId, counter, dest, receiveLength);
			
			/* Send the available numbers i can send */
			MPI_Send(&sendLength, 1, MPI_INT, dest, 2, Communicator);
			//printf("THREAD %d: Iteration %d:  Length sent to dest %d: is %d: 983 \n", myId, counter, dest, sendLength);
			
			/* Calculate the difference between sendLength and receiveLength */
			difference = sendLength - receiveLength;
			
			if( difference >= 0)
				exchangeLength = receiveLength;
			else
				exchangeLength = sendLength;
			
			printf(" I am %d I send data to %d SEND LENGTH: %d RECEIVE LENGTH: %d EXCHANGE LENGTH: %d\n", myId, dest, sendLength, receiveLength, exchangeLength);
			
			if( exchangeLength > 0){
				
				for( j = 0; j < dimensions; j++){
					
					/* receive exchanged data */
					MPI_Recv( &(buffer[receiveIndex]), exchangeLength, MPI_FLOAT, source, 3, Communicator, &Stat);
					//printf("THREAD %d: Iteration %d:  Exchange received from dest %d: is %d: 998 \n", myId, counter, dest, exchangeLength);
					
					/* Swap coordinates to be stored with coordinates to be sent */
					for( i = receiveIndex; i < (receiveIndex + exchangeLength) ; i++){
						temp = ElementsArray[i].coordinates[j];
						ElementsArray[i].coordinates[j] = buffer[i];
						buffer[i] = temp;
					}
					
					/*send exchanged values */
					MPI_Send( &(buffer[receiveIndex]), exchangeLength, MPI_FLOAT, dest, 4, Communicator);
					//printf("THREAD %d: Iteration %d:  Exchange sent to dest %d: is %d: 1003 \n", myId, counter, dest, exchangeLength);
				}
				
				/* Increase receiveIndex */
				receiveIndex = receiveIndex + exchangeLength;
				
				
				if( difference >= 0){ 
					/* If difference is >= 0 then my partner is fine so i will communicate with the next */					
					dest++;
				}
				
				/* update sendLength */
				sendLength = sendLength - exchangeLength;
								
			}
			else /* if i do no exchange then i communicate with the next node */
				dest++;
			
		}
		
		//print( _NODE->array, _NODE->arrayLength);
		/* In case no rearrangement happened */
		if( counter == 0)
			printf("THREAD %d: No rearrangement needed", myId);
		
		/* If i am not the P Node inform the next node to start communication */
		if( myId < noProcesses - 1){
			int next_node = myId + 1;
			printf("THREAD %d: Sending dest to %d\n", myId, next_node);
			MPI_Send(&dest, 1, MPI_INT, next_node, 6, Communicator);
			printf("THREAD %d: dest sent to %d\n", myId, next_node);
		}
		free(buffer);
	}
}

/*** Inform Master about Common Tree ***/
void inform_master( struct SIMPLE_NODE **commonTree, struct NODE *root, int level, int processId){
	int j, i, k, noProcesses, subprocess;
	struct NODE *local_NODE = root;
	subprocess = processId;
	noProcesses = pow(2, level);
	
	for( j = 0; j < level + 1; j++){
		if( processId == 0){
			commonTree[j][0].Vantage_Point.coordinates = (float*)malloc(sizeof(float)*dimensions); 
			for( i = 0; i < dimensions; i++)
				commonTree[j][0].Vantage_Point.coordinates[i] = local_NODE->Vantage_Point.coordinates[i];
			commonTree[j][0].median = local_NODE->median;			
			local_NODE = local_NODE->Left_Node;
			for( i = 1; i < pow(2,j); i++){
				commonTree[j][i].Vantage_Point.coordinates = (float*)malloc(sizeof(float)*dimensions);
				printf("THREAD %d: Ready to receive from %d position [%d][%d]\n", processId, noProcesses*i, j,i);
				for( k = 0; k < dimensions; k++)
					MPI_Recv( &commonTree[j][i].Vantage_Point.coordinates[k], 1, MPI_FLOAT, noProcesses*i, 2*i, MPI_COMM_WORLD, &Stat);
				printf("THREAD %d: received from %d Coordinates\n", processId, noProcesses*i);
				MPI_Recv( &(commonTree[j][i].median), 1, MPI_FLOAT, noProcesses*i, 2*i + 1, MPI_COMM_WORLD, &Stat);
				printf("THREAD %d: received from %d Median\n", processId, noProcesses*i);
			}
			noProcesses = noProcesses/2;
		}
		else{
			
			for( i = 1; i < pow(2,j); i++){
				if( processId == i*noProcesses){
					printf("THREAD %d: Ready to send to %d position [%d][%d]\n", noProcesses*i, processId, j,i);
					for( k = 0; k < dimensions; k++)
						MPI_Send( &(local_NODE->Vantage_Point.coordinates[k]), 1, MPI_FLOAT, 0, 2*i, MPI_COMM_WORLD);
					printf("THREAD %d: sent to %d Coordinates\n", noProcesses*i, processId);
					MPI_Send( &(local_NODE->median), 1, MPI_FLOAT, 0, 2*i + 1, MPI_COMM_WORLD);
					printf("THREAD %d: sent to %d Median\n", noProcesses*i, processId);
				}
			}
			if( subprocess < noProcesses/2){
				subprocess = subprocess;
				local_NODE = local_NODE->Left_Node;
			}
			else{
				subprocess = subprocess - noProcesses/2;			
				local_NODE = local_NODE->Right_Node;
			}	
			noProcesses = noProcesses/2;	
		}
	}
	
}

/*** Find threads first local node ***/
struct NODE *who_is_my_local_Node( int processId, int noProcesses, struct NODE *root){
	struct NODE *return_Node = root;
	while(noProcesses>1){
		if(	processId < noProcesses/2){
			return_Node = return_Node->Left_Node;
			processId = processId;
		}
		else{
			return_Node = return_Node->Right_Node;
			processId = processId - noProcesses/2;
		}
		noProcesses = noProcesses/2;
	}	
	return return_Node;
}

/*** Inform other nodes about Common Tree ***/
void inform_other_nodes( struct SIMPLE_NODE **commonTree, int processId, int level){
	int i, j, k;
	for( j = 0; j < level; j++){
		for( i = 0; i < pow( 2, j); i++){
			if( processId > 0 )
				commonTree[j][i].Vantage_Point.coordinates = (float*)malloc( dimensions * sizeof(float));
			for( k = 0; k < dimensions; k++)
				MPI_Bcast( &(commonTree[j][i].Vantage_Point.coordinates[k]), 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
			MPI_Bcast( &(commonTree[j][i].median), 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
		}
	}
	
}

/*** Partitions the array based on pivot and returns partitioned arrays ***/
void partition (float *array, int elements, float pivot, float **arraysmall, float **arraybig, int *endsmall, int *endbig, struct ELEMENT *ElementsArray)
{
    int right=elements-1;
    int left=0;
    int pos;
	int flag;
    if(elements==1)
    {
        if(pivot>array[0])
        {
            *endsmall=1;  //One value in the small part
            *endbig=0;   //Zero on the big one
            *arraysmall=array;  //There is no big array therefore NULL value
            *arraybig=NULL;			
        }
        else if(pivot<=array[0])
        {
            *endsmall=0;    //The exact opposite of the above actions.
            *endbig=1;
            *arraysmall=NULL;
            *arraybig=array;
        }
    }
    else if(elements>1)
    {	flag = 0;
        while(left<right)
        {
            while(array[left]<pivot)
            {
                left++;
                if(left>=elements)
                {
                    break;
                }
            }
			
            while(array[right]>=pivot)
            {		
				right--;
				if(right<0)
				{
					break;
				}
				
            }
            if(left<right && flag == 0)
            {
                swap_values(array,left,right, ElementsArray);
            }
        }
        pos=right;
        if(pos<0)                   //Arrange the arrays so that they are split into two smaller ones.
        {                               //One containing the small ones. And one the big ones.
            *arraysmall=NULL;           //However these arrays are virtual meaning that we only save the pointer values of the beging and end
        }                               //of the "real" one.
        else
        {
            *arraysmall=array;
        }
        *endsmall=pos+1;
        *arraybig=&array[pos+1];
        *endbig=elements-pos-1;
    }
}

/*** Finds locally median ***/
float selection(float *array,int number, struct ELEMENT *ElementsArray)
{
    float *arraybig;
    float *arraysmall;
    int endsmall=0;
    int endbig=0;
    float *arraytobeused;
    int i;
    int counter=0;
    int k;
    float pivot;
    float median;
    k=(int)number/2 + 1;
    arraytobeused=array;
	struct ELEMENT *ElementsArrayToUse;
	ElementsArrayToUse = ElementsArray;
	
    for(;;)
    {
        pivot=arraytobeused[rand() % number];
        partition(arraytobeused,number,pivot,&arraysmall,&arraybig,&endsmall,&endbig, ElementsArrayToUse);
        if(endbig>k)
        {
            number=endbig;
            arraytobeused=arraybig;
            for(i=0;i<endbig;i++)
            {
                if(pivot==arraybig[i])
                    counter++;
                else
                    break;
            }
            if(counter==endbig)
            {
                median=arraybig[0];
                break;
            }
            else
                counter=0;
            //end of count equals
        }
        else if(endbig<k)
        {
            number=endsmall;
            arraytobeused=arraysmall;
            k=k-endbig;
        }
        else
        {
            median=pivot;
            break;
        }
    }
    return median;
}

/*** This function is used to find first big element position needed for rearrangement and to correct last partition ***/
int correction( float *array, struct ELEMENT *ElementsArray, int length, float pivot){	
	int i, flag = 0;
	int first_big_element;
	for( i = 0; i < length; i++){
		if( array[i] > pivot && flag == 0){
			first_big_element = i;
			flag = 1;
			printf("FOUND: index = %d distance = %f \n", i ,array[i]);
		}
		if( flag == 1 && array[i] <= pivot){			
			swap_values( array, first_big_element, i, ElementsArray);
			first_big_element++;
			printf("New Index %d distance %f\n", first_big_element , array[first_big_element]);
		}
	}
	
	if( flag == 0)
		return length;
		
	return first_big_element;
}	


/*----------KNN Fuctions------*/

void validate_knn( struct ELEMENT *ElementsArray, int partLength, struct knn *VP_knn){
	float *distances = (float*)malloc(partLength*sizeof(float));
	calculate_distances( distances, ElementsArray, partLength, VP_knn->query.coordinates);
	int *indexes = (int*)malloc(VP_knn->k * sizeof(int));
	int i , k = 0;
	float temp = FLT_MAX;
	for( i = 0; i < partLength; i++){
		if(distances[i] != 0){
			if( distances[i] < temp){
				temp = distances[i];
				k = i;
			}
		}
	}
	printf(" %f \n", temp);
	
}
void add_element( int i, struct knn *VP_knn,  float distance, float *coordinates){
	int j;
	float temp = VP_knn->distances[0];
	VP_knn->distances[i] = distance; 
	for(j = 0; j < dimensions; j++)
		VP_knn->Elements[i].coordinates[j] = coordinates[j];
	
	for( j = 0; j < VP_knn->added; j++){
		if( VP_knn->distances[j] > temp){
			VP_knn->largest_distance_index = j;
			temp = VP_knn->distances[j];
		}
	}
	
	if( VP_knn-> added == VP_knn->k - 1)
		VP_knn->t = VP_knn->distances[VP_knn->largest_distance_index];

}
void update_results( float distance, float *coordinates ,struct knn *VP_knn){
	
	if( VP_knn->added == VP_knn->k - 1){
		add_element( VP_knn->largest_distance_index, VP_knn, distance, coordinates);
	}
	else{
		VP_knn->added = VP_knn->added+1;
		add_element( VP_knn->added, VP_knn, distance, coordinates);
	}
}

void search_global_knn( struct knn *VP_knn, struct SIMPLE_NODE **commonTree, int level, int position){
	float distance;
	int processId, i;
	MPI_Comm_rank (MPI_COMM_WORLD, &processId);	
    
	if( level < MAX_level){
		distance = dist(&(VP_knn->query.coordinates[0]), &(commonTree[level][position].Vantage_Point.coordinates[0]));
		
		if( distance < VP_knn->t){
			//printf("[%d]{%d]: Update t: distance = %f < t = %f \n", level, position, distance, VP_knn->t);
			update_results( distance, commonTree[level][position].Vantage_Point.coordinates, VP_knn);
		}
		
		if( distance <= commonTree[level][position].median){
		
			if( distance - VP_knn->t <= commonTree[level][position].median){
				search_global_knn( VP_knn, commonTree, level + 1, 2*position);
			}
			
			if( distance + VP_knn->t >= commonTree[level][position].median){
				search_global_knn( VP_knn, commonTree, level + 1, 2*position + 1);
			}
		}
		else{		
			if( distance + VP_knn->t >= commonTree[level][position].median){
				search_global_knn( VP_knn, commonTree, level + 1, 2*position + 1);
			}
			
			if( distance - VP_knn->t <= commonTree[level][position].median){
				search_global_knn( VP_knn, commonTree, level + 1, 2*position);
			}
		}
	}
	else{
		if( VP_knn->send_array == NULL && processId != position){
			VP_knn->send_array = (int*)malloc( pow(2, MAX_level)*sizeof(int));
			for( i = 0; i < pow(2, MAX_level); i++)
				VP_knn->send_array[i] = 0;				
		}	
		if( VP_knn->send_array != NULL){
			VP_knn->send_array[processId] = -1;	
			if( VP_knn->send_array[position] != -1 && processId != position){			
				/* printf("[%d][%d]: needs further investigation", level, position);
				printf("Upper Node Vantage Point distance from query : %f\n", dist(&(VP_knn->query.coordinates[0]), &(commonTree[level-1][position/2].Vantage_Point.coordinates[0])));
				printf("t is : %f\n", VP_knn->t);
				printf */("Median is : %f\n",commonTree[level-1][position/2].median);
				VP_knn->send_array[position] = 1;		
			}
		}
	}	
} 

void search_knn( struct NODE *root, struct knn *VP_knn){
	float distance;
	if( root == NULL) return;
	int i;
	
	/* Calculate distance between query and NODE */
	distance = dist( VP_knn->query.coordinates, root->Vantage_Point.coordinates);
	
	/* for( i = 0; i < dimensions; i++)
		printf(" %f", root->Vantage_Point.coordinates[i]);
	printf("median : %f ", root->median);
	printf("dist = %f, tau = %f\n", distance, VP_knn->t); */
	
	if( distance < VP_knn->t){
		update_results( distance, root->Vantage_Point.coordinates, VP_knn);
	}
	
	if( root->Left_Node == NULL && root->Right_Node == NULL){
		return;
	}
	
	if( distance <= root->median){
		
		if( distance - VP_knn->t <= root->median){
				search_knn( root->Left_Node, VP_knn);
		}
		
		if( distance + VP_knn->t >= root->median){
			search_knn( root->Right_Node, VP_knn);
		}
	}
	else{		
		if( distance + VP_knn->t >= root->median){
			search_knn( root->Right_Node, VP_knn);
		}
		
		if( distance - VP_knn->t <= root->median){
			search_knn( root->Left_Node, VP_knn);
		}
	}
	
}

void initialize_knn( struct knn *VP_knn, struct ELEMENT Element, int k) {
	int i;
	VP_knn->distances = (float*)malloc(k * sizeof(float));
	VP_knn->distances[0] = 0;
	VP_knn->Elements = ( struct ELEMENT*)malloc(k * sizeof( struct ELEMENT));
	for( i = 0; i < k; i++)
		VP_knn->Elements[i].coordinates = (float*)malloc( dimensions * sizeof(float));
	VP_knn->query.coordinates = (float*)malloc( dimensions * sizeof(float));
	for( i = 0; i < dimensions; i++)
		VP_knn->query.coordinates[i] = Element.coordinates[i];
	VP_knn->t = FLT_MAX;	
	VP_knn->k = k;
	VP_knn->added = -1;
	VP_knn->largest_distance_index = 0;
	VP_knn->send_array = NULL;	
} 


void update_Node( struct NODE *root, struct SIMPLE_NODE **commonTree, int row_level, int fathers_position){
	int i;
	
	if( row_level < MAX_level){
		//printf("--LEVEL %d--\n", row_level);
		if( root->Left_Node == NULL){
			//printf("LEVEL: %d father pos %d LEFT NULL\n",row_level, fathers_position);
			root->Left_Node = (struct NODE*)malloc(1 * sizeof(struct NODE));
			root->Left_Node->Vantage_Point.coordinates = (float*)malloc( dimensions * sizeof(float));
			root->Left_Node->Father_Node = root;
			for( i = 0; i < dimensions; i++)
				root->Left_Node->Vantage_Point.coordinates[i] = commonTree[row_level+1][2*fathers_position].Vantage_Point.coordinates[i];
			root->Left_Node->median = commonTree[row_level + 1][2*fathers_position].median;
			root->Left_Node->Left_Node == NULL;
			root->Left_Node->Right_Node == NULL;
			printf("!");
		}
		if( root->Right_Node == NULL){
			//printf("LEVEL: %d father pos %d RIGHT NULL\n",row_level, fathers_position);
			root->Right_Node = (struct NODE*)malloc(1 * sizeof(struct NODE));
			root->Right_Node->Vantage_Point.coordinates = (float*)malloc( dimensions * sizeof(float));
			root->Right_Node->Father_Node = root;
			for( i = 0; i < dimensions; i++)
				root->Right_Node->Vantage_Point.coordinates[i] = commonTree[row_level+1][2*fathers_position+1].Vantage_Point.coordinates[i];
			root->Right_Node->median = commonTree[row_level+1][2*fathers_position+1].median;
			root->Right_Node->Left_Node == NULL;
			root->Right_Node->Right_Node == NULL;
		}
		
		update_Node( root->Left_Node, commonTree, row_level + 1, 2*fathers_position);
		update_Node( root->Right_Node, commonTree, row_level + 1, 2*fathers_position+1);
	}
}	

void knn( struct NODE *root, struct ELEMENT *ElementsArray, int partLength, struct SIMPLE_NODE **commonTree, int k, struct NODE *local_NODE, struct timeval *fourth, struct timezone *tzp){
	int i, l, j, h, q, noProcesses, processId;
	
	noProcesses = pow(2, MAX_level);
	MPI_Comm_rank (MPI_COMM_WORLD, &processId);	/* get current process id */
	struct knn *VP_knn = (struct knn*)malloc(partLength * sizeof(struct knn));
		
	//MPI_Barrier(MPI_COMM_WORLD);
	//sleep(2);
	int **sendIndex;
	int *sendLengths;
	sendIndex = (int**)malloc( noProcesses * sizeof(int*));	
	sendLengths = (int*)malloc( noProcesses * sizeof(int));
	for( i = 0; i < noProcesses; i++)
		sendLengths[i] = 0;
	
	
	for( l = 0; l < noProcesses; l++){
		if( processId == l){
			for( i = 0; i < partLength; i++){
				initialize_knn( &VP_knn[i], ElementsArray[i], k);
				search_knn( root, &VP_knn[i]);
				search_global_knn( &VP_knn[i], commonTree, 0, 0);	
			}
			
			printf("THREAD %d: 1st knn search completed \n", processId);
			for( i = 0; i < partLength; i++){
				if(VP_knn[i].send_array != NULL){					
					for( j = 0; j < noProcesses; j++){
						if( VP_knn[i].send_array[j] == 1){
							sendLengths[j]++;
							//printf("sendLengths[%d] %d\n", j, sendLengths[j]);
							sendIndex[j] = (int *)realloc( sendIndex[j], sendLengths[j]*sizeof(int));								
							sendIndex[j][sendLengths[j] - 1] = i;	
							break;
						}
					}
				}
			}
		}
		//MPI_Barrier(MPI_COMM_WORLD);
		
	}
	//sleep(1);
	/* for( j = 0; j < noProcesses; j++){
		if(processId == j){
			printf("THREAD:%d", processId);
			for( i = 0; i < noProcesses; i++){
				printf(" %d", sendLengths[i]);
			}
		printf("\n");
			for( i = 0; i < noProcesses; i++){
				printf("SENTD TO %d:", i);
				for( h = 0; h < sendLengths[i]; h++)
					printf(" %d", sendIndex[i][h]);
				printf("\n");
			}
		}
		//sleep(1);
		//MPI_Barrier(MPI_COMM_WORLD);
	} */
	
	
	
	
	for( i = 0; i < noProcesses; i++){
		
		if( i == processId){ /* If a am i. Then i receive controversial elements and i test them. Then i send them back */
			printf("THREAD %d: Waiting Other Threads to send Elements\n", processId);
			/* RECEIVE ELEMENTS FROM EACH ARRAY */
			int *recvLengths = (int*)malloc(sizeof(int)*noProcesses);
			for( j = 0; j < noProcesses; j++){
				
				if( j != i){ /*Receive from anyone except my self */
					
					//1
					MPI_Recv(&(recvLengths[j]), 1, MPI_INT, j, j + 1, MPI_COMM_WORLD, &Stat);
					printf("THREAD %d: Received Length from %d: %d\n", processId, j, recvLengths[j]); 
					
					if( recvLengths[j] > 0){
						/* Receive knn details */
						struct knn *recvKNN = (struct knn*)malloc( recvLengths[j] * sizeof(struct knn));
						for( h = 0; h < recvLengths[j]; h++)
							recvKNN[h].k = k;
						float *buffer = (float*)malloc( recvLengths[j] * sizeof(float));
						int   *int_buffer = (int*)malloc( recvLengths[j] * sizeof(int));
						
						/* Receive query */
						//2
						for( h = 0; h < recvLengths[j]; h++)							
							recvKNN[h].query.coordinates = (float*)malloc( dimensions * sizeof(float));		/* Allocate Space for query */
						for( l = 0; l < dimensions; l++){
							MPI_Recv( buffer, recvLengths[j], MPI_FLOAT, j, j + 1,MPI_COMM_WORLD, &Stat);
							for( h = 0; h < recvLengths[j]; h++)
								recvKNN[h].query.coordinates[l] = buffer[h];								
						}
						
						/*for( h = 0; h < recvLengths[j]; h++){	
							for( l = 0; l < dimensions; l++)							
								printf(" %f",recvKNN[h].query.coordinates[l]);
							printf("\n");
						} */
						
						/* Receive distances array */
						//3
						for( h = 0; h < recvLengths[j]; h++){
							recvKNN[h].distances = (float*)malloc( k * sizeof(float));
							MPI_Recv( &(recvKNN[h].distances[0]), k, MPI_FLOAT, j, (h +1)*(j+1), MPI_COMM_WORLD, &Stat);
							/* for( l = 0; l < k; l++)
								printf(" %f", recvKNN[h].distances[l]);
							printf("\n"); */
						}
						
						/* Receive t */
						//4
						MPI_Recv( buffer, recvLengths[j], MPI_FLOAT, j, j+1, MPI_COMM_WORLD, &Stat);
						for( h = 0; h < recvLengths[j]; h++)
							recvKNN[h].t = buffer[h];
						
						/*Receive added */
						//5
						MPI_Recv( int_buffer, recvLengths[j], MPI_INT, j, j + 1, MPI_COMM_WORLD, &Stat);
						for( h = 0; h < recvLengths[j]; h++)
							recvKNN[h].added = int_buffer[h];
						
						/* Receive Largest distance index */
						//6
						MPI_Recv( int_buffer, recvLengths[j], MPI_INT, j, j + 1, MPI_COMM_WORLD, &Stat);
						for( h = 0; h < recvLengths[j]; h++)
							recvKNN[h].largest_distance_index = int_buffer[h];
								
						
						/* Receive neighbor Elements */
						//7
						for( h = 0; h < recvLengths[j]; h++){
							recvKNN[h].Elements = ( struct ELEMENT*)malloc(k * sizeof( struct ELEMENT));	
							for( l = 0; l < k; l++)
								recvKNN[h].Elements[l].coordinates = (float*)malloc( dimensions * sizeof(float));
						}					
						float *elements_buffer = (float*)malloc(k * sizeof(float));
						for( h = 0; h < recvLengths[j]; h++){
							for( q = 0; q < dimensions; q ++){
								MPI_Recv( elements_buffer, k, MPI_FLOAT, j, (j + 1)*(q+1), MPI_COMM_WORLD, &Stat);
								for( l = 0; l < k; l++){
									recvKNN[h].Elements[l].coordinates[q] = elements_buffer[l];
								}
							}
						}
						
						/* Check for new Neighbors */
						
						for( h = 0; h < recvLengths[j]; h++){
							search_knn(local_NODE, &recvKNN[h]);
						}
						
						/* Send The checked Elements back */
						/* Send Distances array */
						//8
						for( h = 0; h < recvLengths[j]; h++){
							MPI_Send( &(recvKNN[h].distances[0]), k, MPI_FLOAT, j, (j + 1)*(h+1), MPI_COMM_WORLD);
						}
						
						/* Send t */
						//9
						for( h = 0; h < recvLengths[j]; h++)
							buffer[h] = recvKNN[h].t;
						MPI_Send( buffer, recvLengths[j], MPI_FLOAT, j, j+1, MPI_COMM_WORLD);
						
						/* Send added */
						//10
						for( h = 0; h < recvLengths[j]; h++)
							int_buffer[h] = recvKNN[h].added;
						MPI_Send( int_buffer, recvLengths[j], MPI_INT, j, j + 1, MPI_COMM_WORLD);
						
						/* Send Largest Element Index */
						//11
						for( h = 0; h < recvLengths[j]; h++)
							int_buffer[h] = recvKNN[h].largest_distance_index;
						MPI_Send( int_buffer, recvLengths[j], MPI_INT, j, j+1, MPI_COMM_WORLD);
						
						/* Send Neighbor Elements */
						//12
						for( h = 0; h < recvLengths[j]; h++){
							for( q = 0; q < dimensions; q++){
								for( l = 0; l < k; l++)
									elements_buffer[l] = recvKNN[h].Elements[l].coordinates[q];
								MPI_Send( elements_buffer, k, MPI_FLOAT, j, (j+1)*(q+1), MPI_COMM_WORLD);
							}
						}
						free(recvKNN);
						free(buffer);
						free(int_buffer);
						free(elements_buffer);
						
					}
				}
			} 
		}
		else{ /* If i am not the i then i send my controversial elements */
						
			/* sendLength of controversial elements to i */
			//1
			MPI_Send( &sendLengths[i], 1, MPI_INT, i, processId + 1, MPI_COMM_WORLD);
			printf("THREAD %d: Sent to %d: %d\n", processId, i, sendLengths[i]);
			
			if(sendLengths[i] > 0){
				float *buffer = (float*)malloc( sendLengths[i]*sizeof(float));
				int *int_buffer = (int*)malloc( sendLengths[i]*sizeof(int));
				/* Send query */
				for( l = 0; l < dimensions; l++){
					for( h = 0; h < sendLengths[i]; h++)
						buffer[h] = VP_knn[sendIndex[i][h]].query.coordinates[l];
					//2
					MPI_Send( buffer, sendLengths[i], MPI_FLOAT, i, processId + 1, MPI_COMM_WORLD);
				}
				
				/* Send Distances */
				//3
				for( h = 0; h < sendLengths[i]; h++){
					for( l = 0; l < k; l++)
						buffer[l] = VP_knn[sendIndex[i][h]].distances[l];
					MPI_Send( buffer, k, MPI_FLOAT, i, (h+1)*(processId + 1), MPI_COMM_WORLD);
				}
				
				/* Send t */
				for( h = 0; h < sendLengths[i]; h++)
					buffer[h] = VP_knn[sendIndex[i][h]].t;
				//4
				MPI_Send( buffer, sendLengths[i], MPI_FLOAT, i, processId + 1, MPI_COMM_WORLD);
				
				/* Send added */
				//5
				for( h = 0; h < sendLengths[i]; h++)
					int_buffer[h] = VP_knn[sendIndex[i][h]].added;
				MPI_Send( int_buffer, sendLengths[i], MPI_INT, i, processId + 1, MPI_COMM_WORLD);
				
				/* Send Largest distance Index */
				//6
				for( h = 0; h < sendLengths[i]; h++)
					int_buffer[h] = VP_knn[sendIndex[i][h]].largest_distance_index;
				MPI_Send( int_buffer, sendLengths[i], MPI_INT, i, processId + 1, MPI_COMM_WORLD);
				
				/* Send Neighbor Elements */
				//7
				float *elements_buffer = (float*)malloc( sendLengths[i] * sizeof(float));
				for( h = 0; h < sendLengths[i]; h++){
					for( q = 0; q < dimensions; q++){
						for( l = 0; l < k; l++)
							elements_buffer[l] = VP_knn[sendIndex[i][h]].Elements[l].coordinates[q];
						MPI_Send( elements_buffer, k, MPI_FLOAT, i, (processId +1)*(q+1), MPI_COMM_WORLD);
					}
				}
				
				/* Receive Controversial elements Back */
				/* Receive Distances array */
				//8
				for( h = 0; h < sendLengths[i]; h++){
					MPI_Recv( &(VP_knn[sendIndex[i][h]].distances[0]), k, MPI_FLOAT, i, (processId + 1)*(h+1), MPI_COMM_WORLD, &Stat);
				}
				
				/* Receive t */
				//9
				MPI_Recv( buffer, sendLengths[i], MPI_FLOAT, i, processId+1, MPI_COMM_WORLD, &Stat);
				for( h = 0; h < sendLengths[i]; h++)
					VP_knn[sendIndex[i][h]].t = buffer[h];
				
				/* Receive added */
				//10
				MPI_Recv( int_buffer, sendLengths[i], MPI_INT, i, processId + 1, MPI_COMM_WORLD, &Stat);
				for( h = 0; h < sendLengths[i]; h++)
					VP_knn[sendIndex[i][h]].added = int_buffer[h];
				
				/* Receive Largest distance Index */
				//11
				MPI_Recv( int_buffer, sendLengths[i], MPI_INT, i, processId +1, MPI_COMM_WORLD, &Stat);
				for( h = 0; h < sendLengths[i]; h++)
					VP_knn[sendIndex[i][h]].largest_distance_index = int_buffer[h];
				
				/* Receive Elements */
				//12
				for( h = 0; h < sendLengths[i]; h++){
					for( q = 0; q < dimensions; q++){
						MPI_Recv( elements_buffer, k, MPI_FLOAT, i, (q+1)*(processId + 1), MPI_COMM_WORLD, &Stat);
						for( l = 0; l < k; l++)
							VP_knn[sendIndex[i][h]].Elements[l].coordinates[q] = elements_buffer[l];
					}
				}
				
				/* Set send array of each element to -1 */
				for( h = 0; h < sendLengths[i]; h++)
					VP_knn[sendIndex[i][h]].send_array[i] = -1;
				
				free(buffer);
				free(int_buffer);
				free(elements_buffer);
				/** Exchange test has been completed **/
				/** check if further investigation is needed **/	
			}			
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);
	if(processId == 0)
		gettimeofday(fourth, tzp);
		
	/* Knn Problem Completed */
	
	/*** PRINT KNN ***/
	/* for(h = 0; h < noProcesses; h++){
		if( h == processId){
			printf("THREAD %d:Element: Neighbors\n", processId);
			for( i = 0; i < partLength; i++){
				for( j = 0; j < k; j++){
					for( l = 0; l < dimensions; l++){
						printf(" [%f]",VP_knn[i].Elements[j].coordinates[l]);
					}
					printf(",");
				}
				printf("\n");
					
			}
		}
		MPI_Barrier(MPI_COMM_WORLD);
		sleep(1);
	} */
}