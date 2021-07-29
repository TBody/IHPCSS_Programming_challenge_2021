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
    INTEGER :: iteration_count = -1
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
    integer :: e_send_req, w_send_req, e_recv_req, w_recv_req
    real(8), dimension(0:ROWS_MPI, 1:COLS_MPI+1) :: temp_buffer
    REAL(8) :: max_temp_change


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

    DO WHILE (total_time_so_far .LT. MAX_TIME)
        
        IF (my_rank == MASTER_PROCESS_RANK) THEN
            total_time_so_far = MPI_Wtime() - start_time
        END IF

        CALL MPI_IBcast(total_time_so_far, 1, MPI_DOUBLE_PRECISION, MASTER_PROCESS_RANK, cart_comm, bcast_req, ierr)

        IF (MOD(iteration_count, SNAPSHOT_INTERVAL) .EQ. 0) THEN
            max_temp_change = my_temperature_change
            call MPI_ireduce(max_temp_change, global_temperature_change, 1, MPI_DOUBLE_PRECISION, MPI_MAX, &
                             MASTER_PROCESS_RANK, cart_comm, reduce_req, ierr)
            
            ! Verified that the sum of the gather is equal to the sum of the individual sends and recieves 
            call MPI_igather(temp_buffer, ROWS_MPI * COLS_MPI, MPI_DOUBLE_PRECISION, &
                    snapshot, ROWS_MPI * COLS_MPI, MPI_DOUBLE_PRECISION, &
                    MASTER_PROCESS_RANK, cart_comm, gather_req, ierr)

        END IF

        ! Send data to up neighbour for its ghost cells. If my W_rank is MPI_PROC_NULL, this MPI_Ssend will do nothing.
        CALL MPI_Isend(temperatures(0,1), ROWS_PER_MPI_PROCESS, MPI_DOUBLE_PRECISION, W_rank, 101, MPI_COMM_WORLD, W_send_req, ierr)

        ! Receive data from down neighbour to fill our ghost cells. If my E_rank is MPI_PROC_NULL, this MPI_Recv will do nothing.
        CALL MPI_IRecv(temperatures_last(0,COLUMNS_PER_MPI_PROCESS+1), ROWS_PER_MPI_PROCESS, MPI_DOUBLE_PRECISION, E_rank, &
                      101, MPI_COMM_WORLD, E_recv_req, ierr)

        ! Send data to down neighbour for its ghost cells. If my E_rank is MPI_PROC_NULL, this MPI_Ssend will do nothing.
        CALL MPI_Isend(temperatures(0, COLUMNS_PER_MPI_PROCESS), ROWS_PER_MPI_PROCESS, MPI_DOUBLE_PRECISION, E_rank, 102,&
                       MPI_COMM_WORLD, E_send_req, ierr)

        ! Receive data from up neighbour to fill our ghost cells. If my W_rank is MPI_PROC_NULL, this MPI_Recv will do nothing.
        CALL MPI_IRecv(temperatures_last(0,0), ROWS_PER_MPI_PROCESS, MPI_DOUBLE_PRECISION, W_rank, 102, MPI_COMM_WORLD, &
                      W_recv_req, ierr)
        
        call MPI_WAITALL(4, (/ W_send_req, E_send_req, W_recv_req, E_recv_req /), MPI_STATUSES_IGNORE, ierr)

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

        IF (MOD(iteration_count+1, SNAPSHOT_INTERVAL) .EQ. 0) THEN
            my_temperature_change = 0.0
            DO j = 1, COLUMNS_PER_MPI_PROCESS
                DO i = 0, ROWS_PER_MPI_PROCESS - 1
                    my_temperature_change = max(abs(temperatures(i,j) - temperatures_last(i,j)), my_temperature_change)
                END DO
            END DO
            DO j = 1, COLS_MPI + 1
                DO i = 0, ROWS_MPI
                    temp_buffer(i, j) = temperatures(0:ROWS_MPI, 1:COLS_MPI+1)
                ENDDO
            ENDDO
        ENDIF

        DO j = 1, COLUMNS_PER_MPI_PROCESS
            DO i = 0, ROWS_PER_MPI_PROCESS - 1
                temperatures_last(i,j) = temperatures(i,j)
            END DO
        END DO

        IF (MOD(iteration_count, SNAPSHOT_INTERVAL) .EQ. 0) THEN
            CALL MPI_WAITALL(3, (/ reduce_req, bcast_req, gather_req  /), MPI_STATUSES_IGNORE, ierr)
            if (my_rank == MASTER_PROCESS_RANK) then
                WRITE(*,'(A,I0,A,F0.18)') 'Iteration ', iteration_count, ': ', global_temperature_change
                if (print_snap_sum) then
                    WRITE(*,'(A,I10,A,5E14.7)') 'Iteration ', iteration_count, ': sum snapshot: ', sum(snapshot)
                endif
            endif
        else
            CALL MPI_WAIT(bcast_req, MPI_STATUS_IGNORE, ierr)
        END IF

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
