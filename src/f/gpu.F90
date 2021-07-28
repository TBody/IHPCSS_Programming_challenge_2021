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
    !> Rank of my top neighbour if any
    INTEGER :: N_rank
    !> Rank of my bottom neighbour if any
    INTEGER :: S_rank
    !> Array that will contain my part chunk. It will include the 2 ghost rows (1 left, 1 right)
    REAL(8), DIMENSION(1:ROWS_MPI,0:COLS_MPI+1) :: temperatures
    !> Temperatures from the previous iteration, same dimensions as the array above.
    REAL(8), DIMENSION(1:ROWS_MPI,0:COLS_MPI+1) :: temperatures_last
    !> On master process only: contains all temperatures read from input file.
    REAL(8), DIMENSION(0:ROWS-1,0:COLS-1) :: all_temperatures
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
    REAL(8), DIMENSION(1:ROWS,1:COLS) :: snapshot
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
    integer, parameter :: NX = COLS/COLS_MPI
    integer, parameter :: NY = ROWS/ROWS_MPI
    integer, dimension(0:1) :: dims, coords
    logical, parameter :: check_snapshot = .false.
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

    CALL MPI_scatter(all_temperatures, ROWS_MPI * COLS_MPI, MPI_DOUBLE_PRECISION, &
                     temperatures_last(1,1), ROWS_MPI * COLS_MPI, MPI_DOUBLE_PRECISION, MASTER_PROCESS_RANK, &
                     cart_comm, ierr)
    
    ! Copy the temperatures into the current iteration temperature as well
    !$acc data create(temperatures) copyin(temperatures_last)
    !$acc kernels
    DO j = 1, COLS_MPI
        DO i = 1, ROWS_MPI
            temperatures(i,j) = temperatures_last(i,j)
        ENDDO
    ENDDO
    !$acc end kernels
    !$acc update host(temperatures(:,1), temperatures(:,COLS_MPI)) async(1)

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
            
            !$acc update host(temperatures(1:ROWS_MPI, 1:COLS_MPI))
            ! Verified that the sum of the gather is equal to the sum of the individual sends and recieves 
            call MPI_igather(temperatures(1,1), ROWS_MPI * COLS_MPI, MPI_DOUBLE_PRECISION, &
                    snapshot, ROWS_MPI * COLS_MPI, MPI_DOUBLE_PRECISION, &
                    MASTER_PROCESS_RANK, cart_comm, gather_req, ierr)

        END IF
        
        !$acc wait(1)
        CALL MPI_Isend(temperatures(:,1), ROWS_MPI, MPI_DOUBLE_PRECISION, W_rank, &
                       101, cart_comm, W_send_req, ierr)
        CALL MPI_IRecv(temperatures_last(:,COLS_MPI+1), ROWS_MPI, MPI_DOUBLE_PRECISION, E_rank, &
                       101, cart_comm, E_recv_req, ierr)
        CALL MPI_Isend(temperatures(:, COLS_MPI), ROWS_MPI, MPI_DOUBLE_PRECISION, E_rank, &
                       102, cart_comm, E_send_req, ierr)
        CALL MPI_IRecv(temperatures_last(:,0), ROWS_MPI, MPI_DOUBLE_PRECISION, W_rank, &
                       102, cart_comm, W_recv_req, ierr)

        my_temperature_change = 0.0
        !$acc kernels async(2)
        DO j = 2, COLS_MPI - 1
            ! Process all cells between the first and last columns excluded, which each has both left and right neighbours
            DO i = 2, ROWS_MPI - 1
                    temperatures(i,j) = 0.25 * (temperatures_last(i-1,j  ) + &
                                                temperatures_last(i+1,j  ) + &
                                                temperatures_last(i  ,j-1) + &
                                                temperatures_last(i  ,j+1))
            END DO
        END DO
        DO j = 2, COLS_MPI - 1
            ! Process the cell at the first row, which has no up neighbour
            temperatures(1,j) = (temperatures_last(1,j-1) + &
                                 temperatures_last(1,j+1) + &
                                 temperatures_last(2,j  )) * one_third
            ! Process the cell at the bottom row, which has no down neighbour
            temperatures(ROWS_MPI,j) = (temperatures_last(ROWS_MPI, j - 1) + &
                                                    temperatures_last(ROWS_MPI, j + 1) + &
                                                    temperatures_last(ROWS_MPI  -1, j)) * one_third
        ENDDO
        !$acc end kernels

        call MPI_WAITALL(4, (/ W_send_req, E_send_req, W_recv_req, E_recv_req /), MPI_STATUSES_IGNORE, ierr)
        !$acc update device(temperatures_last(:,0), temperatures_last(:,COLS_MPI+1))
        !$acc kernels async(3)
        DO i = 2, ROWS_MPI - 1
            temperatures(i,1) = 0.25 * (temperatures_last(i-1,1) + temperatures_last(i+1,1) &
                                      + temperatures_last(i,1-1) + temperatures_last(i,1+1))
            temperatures(i,COLS_MPI) = 0.25 * (temperatures_last(i-1,COLS_MPI) &
                                                            + temperatures_last(i+1,COLS_MPI) &
                                                            + temperatures_last(i,COLS_MPI-1) &
                                                            + temperatures_last(i,COLS_MPI+1))
        END DO
        
        ! Fill in corners
        temperatures(1,1) = (temperatures_last(1,0) + temperatures_last(1,2) &
                           + temperatures_last(2,1)) * one_third
        
        temperatures(ROWS_MPI,1) = (temperatures_last(ROWS_MPI, 1 - 1) &
                                              + temperatures_last(ROWS_MPI, 1 + 1) &
                                              + temperatures_last(ROWS_MPI-1, 1)) * one_third

        temperatures(1,COLS_MPI) = (temperatures_last(1,COLS_MPI-1) &
                                                 + temperatures_last(1,COLS_MPI+1) &
                                                 + temperatures_last(2,COLS_MPI)) * one_third 

        temperatures(ROWS_MPI,COLS_MPI) = &
                 (temperatures_last(ROWS_MPI, COLS_MPI - 1) &
                + temperatures_last(ROWS_MPI, COLS_MPI + 1) &
                + temperatures_last(ROWS_MPI-1, COLS_MPI)) * one_third
        
        !$acc end kernels 
        
        !$acc kernels wait(2, 3)
        temperatures = merge(temperatures, temperatures_last, temperatures_last /= MAX_TEMPERATURE)
        DO j = 1, COLS_MPI
            DO i = 1, ROWS_MPI
                my_temperature_change = max(abs(temperatures(i,j) - temperatures_last(i,j)), my_temperature_change)
                temperatures_last(i,j) = temperatures(i,j)
            END DO
        END DO
        !$acc end kernels
        !$acc update host(temperatures(:,1), temperatures(:,COLS_MPI)) async(1)

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
