!>
!> @file main.c
!> @brief This file contains the source code of the application to parallelise.
!> @details This application is a classic heat spread simulation.
!> @author Ludovic Capelli

PROGRAM main
    USE util
    USE mpi
    use openacc

    IMPLICIT NONE
   
    !> Used to get the error code returned by MPI routines
    INTEGER :: ierr
    !> Ranks for convenience so that we don't throw raw values all over the code
    INTEGER, PARAMETER :: MASTER_PROCESS_RANK = 0
    !> The rank of the MPI process in charge of this instance
    INTEGER :: my_rank
    !> Number of MPI processes in total, commonly called 'comm_size' for 'communicator size'.
    INTEGER :: comm_size
    !> Rank of the first MPI process
    INTEGER, PARAMETER :: FIRST_PROCESS_RANK = 0
    !> Rank of the last MPI process
    INTEGER :: LAST_PROCESS_RANK
    !> Rank of my left neighbour if any
    INTEGER :: left_neighbour_rank
    !> Rank of my right neighbour if any
    INTEGER :: right_neighbour_rank
    !> Array that will contain my part chunk. It will include the 2 ghost rows (1 left, 1 right)
    REAL(8), DIMENSION(0:ROWS_PER_MPI_PROCESS-1,0:COLUMNS_PER_MPI_PROCESS+1) :: temperatures
    !> Temperatures from the previous iteration, same dimensions as the array above.
    REAL(8), DIMENSION(0:ROWS_PER_MPI_PROCESS-1,0:COLUMNS_PER_MPI_PROCESS+1) :: temperatures_last
    !> On master process only: contains all temperatures read from input file.
    REAL(8), DIMENSION(0:ROWS-1,0:COLUMNS-1) :: all_temperatures
    !> Will contain the entire time elapsed in the timed portion of the code
    REAL(8) :: total_time_so_far = 0.0
    !> Contains the timestamp as measured at the beginning of the timed portion of the code
    REAL(8) :: start_time
    !> Iterator
    INTEGER :: i
    !> Iterator
    INTEGER :: j
    !> Iterator
    INTEGER :: k
    !> Iterator
    INTEGER :: l
    !> Keep track of the current iteration count
    INTEGER :: iteration_count = 0
    !> Maximum temperature change observed across all MPI processes
    REAL(8) :: global_temperature_change
    !> Maximum temperature change for us
    REAL(8) :: my_temperature_change 
    !> Used to store temperatures changes from other MPI processes
    REAL(8) :: subtotal
    !> The last snapshot made
    REAL(8), DIMENSION(0:ROWS-1,0:COLUMNS-1) :: snapshot
    real(8), parameter :: one_third = 1.0_8 / 3.0_8
    integer, dimension(:), allocatable :: send_request, snapshot_request
    integer, dimension(:), allocatable :: sendcounts, recvcounts, displs 
    integer :: ndev, idev
    
    CALL MPI_Init(ierr)
    
    !!$acc declare create(temperature, temperature_last)

    ! /////////////////////////////////////////////////////
    ! ! -- PREPARATION 1: COLLECT USEFUL INFORMATION -- //
    ! /////////////////////////////////////////////////////
    CALL MPI_Comm_rank(MPI_COMM_WORLD, my_rank, ierr)
    
    ndev = acc_get_num_devices(acc_device_nvidia)
    idev = mod(my_rank, ndev)
    call acc_set_device_num(idev, acc_device_nvidia)

    CALL MPI_Comm_size(MPI_COMM_WORLD, comm_size, ierr)
    allocate(send_request(0:comm_size - 1), snapshot_request(0:comm_size - 1))
    allocate(sendcounts(0:comm_size - 1), recvcounts(0:comm_size - 1), displs(0:comm_size - 1))

    do i = 0, comm_size - 1
        sendcounts(i) = ROWS_PER_MPI_PROCESS * COLUMNS_PER_MPI_PROCESS
        displs(i) = (ROWS_PER_MPI_PROCESS * COLUMNS_PER_MPI_PROCESS + 0)*j
    enddo
    
    LAST_PROCESS_RANK = comm_size - 1

    left_neighbour_rank = merge(MPI_PROC_NULL, my_rank - 1, my_rank .EQ. FIRST_PROCESS_RANK)
    
    right_neighbour_rank = merge(MPI_PROC_NULL, my_rank + 1, my_rank .EQ. LAST_PROCESS_RANK)

    ! ////////////////////////////////////////////////////////////////////
    ! ! -- PREPARATION 2: INITIALISE TEMPERATURES ON MASTER PROCESS -- //
    ! ////////////////////////////////////////////////////////////////////

    ! The master MPI process will read a chunk from the file, send it to the corresponding MPI process and repeat until all chunks are read.
    IF (my_rank == MASTER_PROCESS_RANK) THEN
        CALL initialise_temperatures(all_temperatures)
    END IF


    CALL MPI_Barrier(MPI_COMM_WORLD, ierr)
    ! ///////////////////////////////////////////
    ! !     ^                                 //
    ! !    / \                                //
    ! !   / | \    CODE FROM HERE IS TIMED    //
    ! !  /  o  \                              //
    ! ! /_______\                             //
    ! ///////////////////////////////////////////
    
    ! ////////////////////////////////////////////////////////
    ! ! -- TASK 1: DISTRIBUTE DATA TO ALL MPI PROCESSES -- //
    ! ////////////////////////////////////////////////////////
    start_time = MPI_Wtime()

    CALL MPI_scatter(all_temperatures, ROWS_PER_MPI_PROCESS * COLUMNS_PER_MPI_PROCESS, MPI_DOUBLE_PRECISION, &
                     temperatures_last(0, 1), ROWS_PER_MPI_PROCESS * COLUMNS_PER_MPI_PROCESS, MPI_DOUBLE_PRECISION, MASTER_PROCESS_RANK, &
                     MPI_COMM_WORLD, ierr)
    
    ! Copy the temperatures into the current iteration temperature as well
    !$acc data create(temperatures) copyin(temperatures_last)
    !$acc kernels
    DO j = 1, COLUMNS_PER_MPI_PROCESS
        DO i = 0, ROWS_PER_MPI_PROCESS - 1
            temperatures(i,j) = temperatures_last(i,j)
        ENDDO
    ENDDO
    !$acc end kernels

    DO WHILE (total_time_so_far .LT. MAX_TIME)
        ! ////////////////////////////////////////
        ! -- SUBTASK 1: EXCHANGE GHOST CELLS -- //
        ! ////////////////////////////////////////
        
        !$acc update host(temperatures(:,1), temperatures(:,COLUMNS_PER_MPI_PROCESS))

        ! Send data to up neighbour for its ghost cells. If my left_neighbour_rank is MPI_PROC_NULL, this MPI_Ssend will do nothing.
        CALL MPI_Ssend(temperatures(0,1), ROWS_PER_MPI_PROCESS, MPI_DOUBLE_PRECISION, left_neighbour_rank, 0, MPI_COMM_WORLD, ierr)

        ! Receive data from down neighbour to fill our ghost cells. If my right_neighbour_rank is MPI_PROC_NULL, this MPI_Recv will do nothing.
        CALL MPI_Recv(temperatures_last(0,COLUMNS_PER_MPI_PROCESS+1), ROWS_PER_MPI_PROCESS, MPI_DOUBLE_PRECISION, right_neighbour_rank, &
                      MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE, ierr)

        ! Send data to down neighbour for its ghost cells. If my right_neighbour_rank is MPI_PROC_NULL, this MPI_Ssend will do nothing.
        CALL MPI_Ssend(temperatures(0, COLUMNS_PER_MPI_PROCESS), ROWS_PER_MPI_PROCESS, MPI_DOUBLE_PRECISION, right_neighbour_rank, 0,&
                       MPI_COMM_WORLD, ierr)

        ! Receive data from up neighbour to fill our ghost cells. If my left_neighbour_rank is MPI_PROC_NULL, this MPI_Recv will do nothing.
        CALL MPI_Recv(temperatures_last(0,0), ROWS_PER_MPI_PROCESS, MPI_DOUBLE_PRECISION, left_neighbour_rank, MPI_ANY_TAG, MPI_COMM_WORLD, &
                      MPI_STATUS_IGNORE, ierr)
        
        !$acc update device(temperatures_last(:,0), temperatures_last(:,COLUMNS_PER_MPI_PROCESS+1))

        ! /////////////////////////////////////////////
        ! // -- SUBTASK 2: PROPAGATE TEMPERATURES -- //
        ! /////////////////////////////////////////////
        my_temperature_change = 0.0
        !$acc kernels
        DO j = 1, COLUMNS_PER_MPI_PROCESS 
            ! Process the cell at the first row, which has no up neighbour
                temperatures(0,j) = (temperatures_last(0,j-1) + &
                                     temperatures_last(0,j+1) + &
                                     temperatures_last(1,j  )) * one_third 
            ! Process all cells between the first and last columns excluded, which each has both left and right neighbours
            DO i = 1, ROWS_PER_MPI_PROCESS - 2
                    temperatures(i,j) = 0.25 * (temperatures_last(i-1,j  ) + &
                                                temperatures_last(i+1,j  ) + &
                                                temperatures_last(i  ,j-1) + &
                                                temperatures_last(i  ,j+1))
            END DO
            ! Process the cell at the bottom row, which has no down neighbour
                temperatures(ROWS_PER_MPI_PROCESS-1,j) = (temperatures_last(ROWS_PER_MPI_PROCESS-1, j - 1) + &
                                                          temperatures_last(ROWS_PER_MPI_PROCESS-1, j + 1) + &
                                                          temperatures_last(ROWS_PER_MPI_PROCESS-2, j)) * one_third
        END DO
        temperatures = merge(temperatures, temperatures_last, temperatures_last /= MAX_TEMPERATURE)
        DO j = 1, COLUMNS_PER_MPI_PROCESS
            DO i = 0, ROWS_PER_MPI_PROCESS - 1
                my_temperature_change = max(abs(temperatures(i,j) - temperatures_last(i,j)), my_temperature_change)
                temperatures_last(i,j) = temperatures(i,j)
            END DO
        END DO
        !$acc end kernels

        ! ///////////////////////////////////
        ! // -- SUBTASK 6: GET SNAPSHOT -- //
        ! ///////////////////////////////////
        IF (MOD(iteration_count, SNAPSHOT_INTERVAL) .EQ. 0) THEN
            call MPI_ireduce(my_temperature_change, global_temperature_change, 1, MPI_DOUBLE_PRECISION, MPI_MAX, &
                             MASTER_PROCESS_RANK, MPI_COMM_WORLD, snapshot_request(0), ierr)
            
            IF (my_rank == MASTER_PROCESS_RANK) THEN
                
                ! Copy locally my own temperature array in the global one
                !$acc kernels copyout(snapshot(0:ROWS_PER_MPI_PROCESS-1, 0:COLUMNS_PER_MPI_PROCESS-1)) async(2)
                DO k = 0, ROWS_PER_MPI_PROCESS-1
                    DO l = 0, COLUMNS_PER_MPI_PROCESS-1
                        snapshot(k,l) = temperatures(k + 1,l)
                    END DO
                END DO
                !$acc end kernels
                
                DO j = 1, comm_size-1
                     CALL MPI_Recv(snapshot(0, j * COLUMNS_PER_MPI_PROCESS), ROWS_PER_MPI_PROCESS * COLUMNS_PER_MPI_PROCESS, &
                                    MPI_DOUBLE_PRECISION, j, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE, ierr)
                ENDDO
                !$acc wait(2)
                CALL MPI_WAIT(snapshot_request(0), MPI_STATUS_IGNORE, ierr)
                !CALL MPI_WAITALL(comm_size, snapshot_request, MPI_STATUSES_IGNORE, ierr)

                WRITE(*,'(A,I0,A,F0.18)') 'Iteration ', iteration_count, ': ', global_temperature_change
            ELSE
                ! Send my array to the master MPI process
                CALL MPI_Ssend(temperatures(0,1), ROWS_PER_MPI_PROCESS * COLUMNS_PER_MPI_PROCESS, MPI_DOUBLE_PRECISION, MASTER_PROCESS_RANK, &
                               0, MPI_COMM_WORLD, ierr) 
            END IF
        END IF

        ! Calculate the total time spent processing
        IF (my_rank == MASTER_PROCESS_RANK) THEN
            total_time_so_far = MPI_Wtime() - start_time
        END IF

        ! Send total timer to everybody so they too can exit the loop if more than the allowed runtime has elapsed already
        CALL MPI_Bcast(total_time_so_far, 1, MPI_DOUBLE_PRECISION, MASTER_PROCESS_RANK, MPI_COMM_WORLD, ierr)

        ! Update the iteration number
        iteration_count = iteration_count + 1
    END DO
    !$acc end data

    ! ///////////////////////////////////////////////
    ! //     ^                                     //
    ! //    / \                                    //
    ! //   / | \    CODE FROM HERE IS NOT TIMED    //
    ! //  /  o  \                                  //
    ! // /_______\                                 //
    ! ///////////////////////////////////////////////

    ! /////////////////////////////////////////
    ! // -- FINALISATION 2: PRINT SUMMARY -- //
    ! /////////////////////////////////////////
    IF (my_rank == MASTER_PROCESS_RANK) THEN
        WRITE(*,'(A,F0.2,A,I0,A)') 'The program took ', total_time_so_far, ' seconds in total and executed ', iteration_count, &
                                   ' iterations.'
    END IF

    CALL MPI_Finalize(ierr)
END PROGRAM main
