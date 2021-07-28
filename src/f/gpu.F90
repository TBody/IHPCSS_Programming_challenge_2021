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
    INTEGER :: W_rank
    !> Rank of my right neighbour if any
    INTEGER :: E_rank
    !> Rank of my top neighbour if any
    INTEGER :: N_rank
    !> Rank of my bottom neighbour if any
    INTEGER :: S_rank
    !> Array that will contain my part chunk. It will include the 2 ghost rows (1 left, 1 right)
    REAL(8), DIMENSION(1:ROWS_PER_MPI_PROCESS,0:COLUMNS_PER_MPI_PROCESS+1) :: temperatures
    !> Temperatures from the previous iteration, same dimensions as the array above.
    REAL(8), DIMENSION(1:ROWS_PER_MPI_PROCESS,0:COLUMNS_PER_MPI_PROCESS+1) :: temperatures_last
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
    real(8), parameter :: one_third = 1.0_8 / 3.0_8
    integer :: ndev, idev
    integer :: reduce_req, gather_req, bcast_req
    integer :: N_send_req, S_send_req, E_send_req, W_send_req
    integer :: N_recv_req, S_recv_req, E_recv_req, W_recv_req
    integer :: cart_comm
    !> Directions (north, south, east, west)
    integer, parameter :: N = 11, S = 12, E = 13, W = 14
    !> Dimensions (x and y)
    integer, parameter :: XDIM = 0, YDIM = 1
    !> Dimension sizes
    integer, parameter :: NX = COLUMNS/COLUMNS_PER_MPI_PROCESS
    integer, parameter :: NY = ROWS/ROWS_PER_MPI_PROCESS
    integer, dimension(0:1) :: dims, coords
    logical, parameter :: check_snapshot = .true.
    CALL MPI_Init(ierr)
    
    ! /////////////////////////////////////////////////////
    ! ! -- PREPARATION 1: COLLECT USEFUL INFORMATION -- //
    ! /////////////////////////////////////////////////////
    dims = (/ NX, NY /)
    CALL MPI_Comm_size(MPI_COMM_WORLD, comm_size, ierr)
    
    if (.not.(dims(0) * dims(1) == comm_size)) then
        print *, "Error: could not decompose MPI with given ROWS_PER_MPI_PROCESS and COLUMNS_PER_MPI_PROCESS", NX, NY, comm_size
    endif

    CALL MPI_cart_create(MPI_COMM_WORLD, 2, dims, &
                         (/ .false., .false./), .false., cart_comm, ierr)

    CALL MPI_Comm_rank(cart_comm, my_rank, ierr)
    
    ndev = acc_get_num_devices(acc_device_nvidia)
    idev = mod(my_rank, ndev)
    call acc_set_device_num(idev, acc_device_nvidia)

    call MPI_cart_coords(cart_comm, my_rank, 2, coords, ierr)
    call MPI_CART_SHIFT(cart_comm, XDIM, +1, W_rank, E_rank, ierr)
    call MPI_CART_SHIFT(cart_comm, YDIM, +1, S_rank, N_rank, ierr)

    ! ////////////////////////////////////////////////////////////////////
    ! ! -- PREPARATION 2: INITIALISE TEMPERATURES ON MASTER PROCESS -- //
    ! ////////////////////////////////////////////////////////////////////

    ! The master MPI process will read a chunk from the file, send it to the corresponding MPI process and repeat until all chunks are read.
    IF (my_rank == MASTER_PROCESS_RANK) THEN
        CALL initialise_temperatures(all_temperatures)
    END IF

    CALL MPI_Barrier(cart_comm, ierr)
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
                     temperatures_last(1,1), ROWS_PER_MPI_PROCESS * COLUMNS_PER_MPI_PROCESS, MPI_DOUBLE_PRECISION, MASTER_PROCESS_RANK, &
                     cart_comm, ierr)
    
    ! Copy the temperatures into the current iteration temperature as well
    !$acc data create(temperatures) copyin(temperatures_last)
    !$acc kernels
    DO j = 1, COLUMNS_PER_MPI_PROCESS
        DO i = 1, ROWS_PER_MPI_PROCESS
            temperatures(i,j) = temperatures_last(i,j)
        ENDDO
    ENDDO
    !$acc end kernels
    !$acc update host(temperatures(:,1), temperatures(:,COLUMNS_PER_MPI_PROCESS)) async(1)

    DO WHILE (total_time_so_far .LT. MAX_TIME)

        ! Calculate the total time spent processing
        IF (my_rank == MASTER_PROCESS_RANK) THEN
            total_time_so_far = MPI_Wtime() - start_time
        END IF

        ! Send total timer to everybody so they too can exit the loop if more than the allowed runtime has elapsed already
        CALL MPI_IBcast(total_time_so_far, 1, MPI_DOUBLE_PRECISION, MASTER_PROCESS_RANK, cart_comm, bcast_req, ierr)

        IF (MOD(iteration_count, SNAPSHOT_INTERVAL) .EQ. 0) THEN
            call MPI_ireduce(my_temperature_change, global_temperature_change, 1, MPI_DOUBLE_PRECISION, MPI_MAX, &
                             MASTER_PROCESS_RANK, cart_comm, reduce_req, ierr)
            
            !$acc update host(temperatures(1:ROWS_PER_MPI_PROCESS, 1:COLUMNS_PER_MPI_PROCESS))
            ! Verified that the sum of the gather is equal to the sum of the individual sends and recieves 
            call MPI_igather(temperatures(1,1), ROWS_PER_MPI_PROCESS * COLUMNS_PER_MPI_PROCESS, MPI_DOUBLE_PRECISION, &
                    snapshot, ROWS_PER_MPI_PROCESS * COLUMNS_PER_MPI_PROCESS, MPI_DOUBLE_PRECISION, &
                    MASTER_PROCESS_RANK, cart_comm, gather_req, ierr)

        END IF
        
        !$acc wait(1)
        ! Send data to up neighbour for its ghost cells. If my W_rank is MPI_PROC_NULL, this MPI_Ssend will do nothing.
        CALL MPI_Isend(temperatures(:,1), ROWS_PER_MPI_PROCESS, MPI_DOUBLE_PRECISION, W_rank, &
                       101, cart_comm, W_send_req, ierr)

        ! Receive data from down neighbour to fill our ghost cells. If my E_rank is MPI_PROC_NULL, this MPI_Recv will do nothing.
        CALL MPI_IRecv(temperatures_last(:,COLUMNS_PER_MPI_PROCESS+1), ROWS_PER_MPI_PROCESS, MPI_DOUBLE_PRECISION, E_rank, &
                       101, cart_comm, E_recv_req, ierr)

        ! Send data to down neighbour for its ghost cells. If my E_rank is MPI_PROC_NULL, this MPI_Ssend will do nothing.
        CALL MPI_Isend(temperatures(:, COLUMNS_PER_MPI_PROCESS), ROWS_PER_MPI_PROCESS, MPI_DOUBLE_PRECISION, E_rank, &
                       102, cart_comm, E_send_req, ierr)

        ! Receive data from up neighbour to fill our ghost cells. If my W_rank is MPI_PROC_NULL, this MPI_Recv will do nothing.
        CALL MPI_IRecv(temperatures_last(:,0), ROWS_PER_MPI_PROCESS, MPI_DOUBLE_PRECISION, W_rank, &
                       102, cart_comm, W_recv_req, ierr)
        
        my_temperature_change = 0.0
        !$acc kernels async(2)
        DO j = 2, COLUMNS_PER_MPI_PROCESS - 1
            ! Process all cells between the first and last columns excluded, which each has both left and right neighbours
            DO i = 2, ROWS_PER_MPI_PROCESS - 1
                    temperatures(i,j) = 0.25 * (temperatures_last(i-1,j  ) + &
                                                temperatures_last(i+1,j  ) + &
                                                temperatures_last(i  ,j-1) + &
                                                temperatures_last(i  ,j+1))
            END DO
        END DO
        DO j = 2, COLUMNS_PER_MPI_PROCESS - 1
            ! Process the cell at the first row, which has no up neighbour
            temperatures(1,j) = (temperatures_last(1,j-1) + &
                                 temperatures_last(1,j+1) + &
                                 temperatures_last(2,j  )) * one_third
            ! Process the cell at the bottom row, which has no down neighbour
            temperatures(ROWS_PER_MPI_PROCESS,j) = (temperatures_last(ROWS_PER_MPI_PROCESS, j - 1) + &
                                                    temperatures_last(ROWS_PER_MPI_PROCESS, j + 1) + &
                                                    temperatures_last(ROWS_PER_MPI_PROCESS  -1, j)) * one_third
        ENDDO
        !$acc end kernels

        call MPI_WAITALL(4, (/ W_send_req, E_send_req, W_recv_req, E_recv_req /), MPI_STATUSES_IGNORE, ierr)
        !$acc update device(temperatures_last(:,0), temperatures_last(:,COLUMNS_PER_MPI_PROCESS+1))
        !$acc kernels async(3)
        DO i = 2, ROWS_PER_MPI_PROCESS - 1
            temperatures(i,1) = 0.25 * (temperatures_last(i-1,1) + temperatures_last(i+1,1) &
                                      + temperatures_last(i,1-1) + temperatures_last(i,1+1))
            temperatures(i,COLUMNS_PER_MPI_PROCESS) = 0.25 * (temperatures_last(i-1,COLUMNS_PER_MPI_PROCESS) &
                                                            + temperatures_last(i+1,COLUMNS_PER_MPI_PROCESS) &
                                                            + temperatures_last(i,COLUMNS_PER_MPI_PROCESS-1) &
                                                            + temperatures_last(i,COLUMNS_PER_MPI_PROCESS+1))
        END DO
        
        ! Fill in corners
        temperatures(1,1) = (temperatures_last(1,0) + temperatures_last(1,2) &
                           + temperatures_last(2,1)) * one_third
        
        temperatures(ROWS_PER_MPI_PROCESS,1) = (temperatures_last(ROWS_PER_MPI_PROCESS, 1 - 1) &
                                              + temperatures_last(ROWS_PER_MPI_PROCESS, 1 + 1) &
                                              + temperatures_last(ROWS_PER_MPI_PROCESS-1, 1)) * one_third

        temperatures(1,COLUMNS_PER_MPI_PROCESS) = (temperatures_last(1,COLUMNS_PER_MPI_PROCESS-1) &
                                                 + temperatures_last(1,COLUMNS_PER_MPI_PROCESS+1) &
                                                 + temperatures_last(2,COLUMNS_PER_MPI_PROCESS)) * one_third 

        temperatures(ROWS_PER_MPI_PROCESS,COLUMNS_PER_MPI_PROCESS) = &
                 (temperatures_last(ROWS_PER_MPI_PROCESS, COLUMNS_PER_MPI_PROCESS - 1) &
                + temperatures_last(ROWS_PER_MPI_PROCESS, COLUMNS_PER_MPI_PROCESS + 1) &
                + temperatures_last(ROWS_PER_MPI_PROCESS-1, COLUMNS_PER_MPI_PROCESS)) * one_third
        
        !$acc end kernels 
        
        !$acc kernels wait(2, 3)
        temperatures = merge(temperatures, temperatures_last, temperatures_last /= MAX_TEMPERATURE)
        DO j = 1, COLUMNS_PER_MPI_PROCESS
            DO i = 1, ROWS_PER_MPI_PROCESS
                my_temperature_change = max(abs(temperatures(i,j) - temperatures_last(i,j)), my_temperature_change)
                temperatures_last(i,j) = temperatures(i,j)
            END DO
        END DO
        !$acc end kernels
        !$acc update host(temperatures(:,1), temperatures(:,COLUMNS_PER_MPI_PROCESS)) async(1)

        IF (MOD(iteration_count, SNAPSHOT_INTERVAL) .EQ. 0) THEN
            CALL MPI_WAITALL(3, (/ reduce_req, bcast_req, gather_req  /), MPI_STATUSES_IGNORE, ierr)
            if (my_rank == MASTER_PROCESS_RANK) then
                WRITE(*,'(A,I0,A,F0.18)') 'Iteration ', iteration_count, ': ', global_temperature_change
                if (check_snapshot) then
                    WRITE(*,'(A,I10,A,5E14.7)') 'Iteration ', iteration_count, ': sum snapshot: ', sum(snapshot)
                endif
            endif
        else
            CALL MPI_WAIT(bcast_req, MPI_STATUS_IGNORE, ierr)
        END IF

        IF (MOD(iteration_count, SNAPSHOT_INTERVAL) .EQ. 0) THEN
            !$acc update host(temperatures)
            IF (my_rank == MASTER_PROCESS_RANK) THEN
                DO j = 0, comm_size-1
                    IF (j .EQ. my_rank) THEN
                        ! Copy locally my own temperature array in the global one
                        DO k = 0, ROWS_PER_MPI_PROCESS-1
                            DO l = 0, COLUMNS_PER_MPI_PROCESS-1
                                snapshot(j * ROWS_PER_MPI_PROCESS + k,l) = temperatures(k + 1,l)
                            END DO
                        END DO
                    ELSE
                        CALL MPI_Recv(snapshot(0, j * COLUMNS_PER_MPI_PROCESS), ROWS_PER_MPI_PROCESS * COLUMNS_PER_MPI_PROCESS, &
                                      MPI_DOUBLE_PRECISION, j, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE, ierr)
                    END IF
                END DO

                WRITE(*,'(A,I0,A,F0.18)') 'Iteration ', iteration_count, ': ', global_temperature_change
                if (check_snapshot) then
                    WRITE(*,'(A,I0,A,5E14.7)'), 'Iteration ', iteration_count, ': sum snapshot: ', sum(snapshot)
                endif
            ELSE
                ! Send my array to the master MPI process
                CALL MPI_Ssend(temperatures(0,1), ROWS_PER_MPI_PROCESS * COLUMNS_PER_MPI_PROCESS, MPI_DOUBLE_PRECISION, MASTER_PROCESS_RANK, &
                               0, MPI_COMM_WORLD, ierr) 
            END IF
        END IF

        ! Update the iteration number (This seems like cheating: we calculated the total time at the start of the step,
        ! so we could get an iteration for free. But to adjust for that (and still have the right output), the inital value
        ! of iteration_count was set to -1)
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
