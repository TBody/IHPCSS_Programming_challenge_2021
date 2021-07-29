!>
!> @file main.c
!> @brief This file contains the source code of the application to parallelise.
!> @details This application is a classic heat spread simulation.
!> @author Ludovic Capelli

PROGRAM main
    USE util
    USE mpi
    USE openacc

    IMPLICIT NONE
    integer, parameter :: COLS = COLUMNS, ROWS_MPI = ROWS_PER_MPI_PROCESS, COLS_MPI = COLUMNS_PER_MPI_PROCESS
   
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
    INTEGER :: W_rank
    !> Rank of my right neighbour if any
    INTEGER :: E_rank
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
    integer :: cart_comm
    logical, parameter :: print_snap_sum = .true.

    integer, parameter :: NX = COLS/COLS_MPI
    integer, parameter :: NY = ROWS/ROWS_MPI
    integer, dimension(0:1) :: dims, coords
    integer, parameter :: XDIM = 0
    integer :: ndev, idev
    real(8), parameter :: one_third = 1.0_8 / 3.0_8
    integer :: reduce_req, gather_req, bcast_req
    real(8) :: sum_snap
    real(8), dimension(0:ROWS_MPI, 1:COLS_MPI+1) :: temp_buffer

    CALL MPI_Init(ierr)
    ! /////////////////////////////////////////////////////
    ! ! -- PREPARATION 1: COLLECT USEFUL INFORMATION -- //
    ! /////////////////////////////////////////////////////
    
    dims = (/ NX, NY /)
    CALL MPI_Comm_size(MPI_COMM_WORLD, comm_size, ierr)
    
    if (.not.(dims(0) * dims(1) == comm_size)) then
        print *, "Error: could not decompose MPI with given ROWS_MPI and COLS_MPI", NX, NY, comm_size
    endif

    CALL MPI_cart_create(MPI_COMM_WORLD, 2, dims, &
                         (/ .false., .false./), .false., cart_comm, ierr)

    CALL MPI_Comm_rank(cart_comm, my_rank, ierr)
    
    ndev = acc_get_num_devices(acc_device_nvidia)
    idev = mod(my_rank, ndev)
    call acc_set_device_num(idev, acc_device_nvidia)

    call MPI_cart_coords(cart_comm, my_rank, 2, coords, ierr)
    call MPI_CART_SHIFT(cart_comm, XDIM, +1, W_rank, E_rank, ierr)

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

    CALL MPI_scatter(all_temperatures, ROWS_MPI * COLS_MPI, MPI_DOUBLE_PRECISION, &
                     temperatures_last(0,1), ROWS_MPI * COLS_MPI, MPI_DOUBLE_PRECISION, MASTER_PROCESS_RANK, &
                     cart_comm, ierr)

    ! Copy the temperatures into the current iteration temperature as well
    DO j = 1, COLUMNS_PER_MPI_PROCESS
        DO i = 0, ROWS_PER_MPI_PROCESS - 1
            temperatures(i,j) = temperatures_last(i,j)
        ENDDO
    ENDDO

    ! /////////////////////////////
    ! // TASK 2: DATA PROCESSING //
    ! /////////////////////////////

    DO WHILE (total_time_so_far .LT. MAX_TIME)
        ! ////////////////////////////////////////
        ! -- SUBTASK 1: EXCHANGE GHOST CELLS -- //
        ! ////////////////////////////////////////

        ! Send data to up neighbour for its ghost cells. If my W_rank is MPI_PROC_NULL, this MPI_Ssend will do nothing.
        CALL MPI_Ssend(temperatures(0,1), ROWS_PER_MPI_PROCESS, MPI_DOUBLE_PRECISION, W_rank, 0, MPI_COMM_WORLD, ierr)

        ! Receive data from down neighbour to fill our ghost cells. If my E_rank is MPI_PROC_NULL, this MPI_Recv will do nothing.
        CALL MPI_Recv(temperatures_last(0,COLUMNS_PER_MPI_PROCESS+1), ROWS_PER_MPI_PROCESS, MPI_DOUBLE_PRECISION, E_rank, &
                      MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE, ierr)

        ! Send data to down neighbour for its ghost cells. If my E_rank is MPI_PROC_NULL, this MPI_Ssend will do nothing.
        CALL MPI_Ssend(temperatures(0, COLUMNS_PER_MPI_PROCESS), ROWS_PER_MPI_PROCESS, MPI_DOUBLE_PRECISION, E_rank, 0,&
                       MPI_COMM_WORLD, ierr)

        ! Receive data from up neighbour to fill our ghost cells. If my W_rank is MPI_PROC_NULL, this MPI_Recv will do nothing.
        CALL MPI_Recv(temperatures_last(0,0), ROWS_PER_MPI_PROCESS, MPI_DOUBLE_PRECISION, W_rank, MPI_ANY_TAG, MPI_COMM_WORLD, &
                      MPI_STATUS_IGNORE, ierr)

        ! /////////////////////////////////////////////
        ! // -- SUBTASK 2: PROPAGATE TEMPERATURES -- //
        ! /////////////////////////////////////////////
        DO j = 1, COLUMNS_PER_MPI_PROCESS 
            ! Process the cell at the first row, which has no up neighbour
            IF (temperatures(0,j) .NE. MAX_TEMPERATURE) THEN
                temperatures(0,j) = (temperatures_last(0,j-1) + &
                                     temperatures_last(0,j+1) + &
                                     temperatures_last(1,j  )) / 3.0
            END IF
            ! Process all cells between the first and last columns excluded, which each has both left and right neighbours
            DO i = 1, ROWS_PER_MPI_PROCESS - 2
                IF (temperatures(i,j) .NE. MAX_TEMPERATURE) THEN
                    temperatures(i,j) = 0.25 * (temperatures_last(i-1,j  ) + &
                                                temperatures_last(i+1,j  ) + &
                                                temperatures_last(i  ,j-1) + &
                                                temperatures_last(i  ,j+1))
                END IF
            END DO
            ! Process the cell at the bottom row, which has no down neighbour
            IF (temperatures(ROWS_PER_MPI_PROCESS-1,j) .NE. MAX_TEMPERATURE) THEN
                temperatures(ROWS_PER_MPI_PROCESS-1,j) = (temperatures_last(ROWS_PER_MPI_PROCESS-1, j - 1) + &
                                                          temperatures_last(ROWS_PER_MPI_PROCESS-1, j + 1) + &
                                                          temperatures_last(ROWS_PER_MPI_PROCESS-2, j)) / 3.0
            END IF
        END DO

        ! ///////////////////////////////////////////////////////
        ! // -- SUBTASK 3: CALCULATE MAX TEMPERATURE CHANGE -- //
        ! ///////////////////////////////////////////////////////
        my_temperature_change = 0.0
        DO j = 1, COLUMNS_PER_MPI_PROCESS
            DO i = 0, ROWS_PER_MPI_PROCESS - 1
                 my_temperature_change = max(abs(temperatures(i,j) - temperatures_last(i,j)), my_temperature_change)
            END DO
        END DO

        ! //////////////////////////////////////////////////////////
        ! // -- SUBTASK 4: FIND MAX TEMPERATURE CHANGE OVERALL -- //
        ! //////////////////////////////////////////////////////////
        call MPI_ireduce(my_temperature_change, global_temperature_change, 1, MPI_DOUBLE_PRECISION, MPI_MAX, &
                             MASTER_PROCESS_RANK, cart_comm, reduce_req, ierr)
        CALL MPI_WAIT(reduce_req, MPI_STATUS_IGNORE, ierr)

        ! //////////////////////////////////////////////////
        ! // -- SUBTASK 5: UPDATE LAST ITERATION ARRAY -- //
        ! //////////////////////////////////////////////////
        DO j = 1, COLUMNS_PER_MPI_PROCESS
            DO i = 0, ROWS_PER_MPI_PROCESS - 1
                temperatures_last(i,j) = temperatures(i,j)
            END DO
        END DO

        ! ///////////////////////////////////
        ! // -- SUBTASK 6: GET SNAPSHOT -- //
        ! ///////////////////////////////////
        IF (MOD(iteration_count, SNAPSHOT_INTERVAL) .EQ. 0) THEN

            temp_buffer = temperatures(0:ROWS_MPI, 1:COLS_MPI+1)
            
            call MPI_gather(temp_buffer, ROWS_MPI * COLS_MPI, MPI_DOUBLE_PRECISION, &
                    snapshot, ROWS_MPI * COLS_MPI, MPI_DOUBLE_PRECISION, &
                    MASTER_PROCESS_RANK, cart_comm, ierr)

            IF (my_rank == MASTER_PROCESS_RANK) THEN
                ! DO j = 0, comm_size-1
                    ! IF (j .EQ. my_rank) THEN
                        ! Copy locally my own temperature array in the global one
                DO k = 0, ROWS_PER_MPI_PROCESS-1
                    DO l = 0, COLUMNS_PER_MPI_PROCESS-1
                        snapshot(k,l) = temperatures(k + 1,l)
                    END DO
                END DO
                    ! ELSE
                        ! CALL MPI_Recv(snapshot(0, j * COLUMNS_PER_MPI_PROCESS), ROWS_PER_MPI_PROCESS * COLUMNS_PER_MPI_PROCESS, &
                                    !   MPI_DOUBLE_PRECISION, j, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE, ierr)
                    ! END IF
                ! END DO

                WRITE(*,'(A,I0,A,F0.18)') 'Iteration ', iteration_count, ': ', global_temperature_change
                if (print_snap_sum) then
                  sum_snap = 0.0
                  do i = 0, COLUMNS - 1
                    do j = 0, ROWS - 1
                      sum_snap = sum_snap + snapshot(i, j)
                    enddo
                  enddo
                  WRITE(*,'(A,I0,A,5E18.10)') 'Iter-snap-sum ', iteration_count, ': ', sum_snap
                endif
            ! ELSE
            !     ! Send my array to the master MPI process
            !     CALL MPI_Ssend(temp_buffer, ROWS_PER_MPI_PROCESS * COLUMNS_PER_MPI_PROCESS, MPI_DOUBLE_PRECISION, MASTER_PROCESS_RANK, &
            !                    0, MPI_COMM_WORLD, ierr) 
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
