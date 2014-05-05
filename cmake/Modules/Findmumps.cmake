#
#  MUMPS_FOUND - system has mumps lib with correct version
#  MUMPS_INCLUDE_DIRS - the mumps include directory
#  MUMPS_LIBRARIES   - the mumps library.

# Copyright (c) 2006, 2007 Montel Laurent, <montel@kde.org>
# Copyright (c) 2008, 2009 Gael Guennebaud, <g.gael@free.fr>
# Copyright (c) 2009 Benoit Jacob <jacob.benoit.1@gmail.com>
# Redistribution and use is allowed according to the terms of the 2-clause BSD license.

# find out the size of a pointer. this is required to only search for
# libraries in the directories relevant for the architecture
if (CMAKE_SIZEOF_VOID_P)
  math (EXPR _BITS "8 * ${CMAKE_SIZEOF_VOID_P}")
endif (CMAKE_SIZEOF_VOID_P)

if (mumps_ROOT)
 set (MUMPS_ROOT "${mumps_ROOT}")
endif (mumps_ROOT)

if (NOT MUMPS_INCLUDE_DIR)
  if (MUMPS_ROOT)
	find_path (MUMPS_INCLUDE_DIR
	  NAMES "zmumps_c.h"
	  PATHS ${MUMPS_ROOT}
	  PATH_SUFFIXES "MUMPS_4.10.0" "include"
	  NO_DEFAULT_PATH
	  )
  endif (MUMPS_ROOT)
endif (NOT MUMPS_INCLUDE_DIR)

if(NOT MUMPS_INCLUDE_DIR)
  message(STATUS "Directory with the MUMPS include not found")
  return()
endif() 

# look for actual mumps library
if (NOT MUMPS_LIBRARY)
  find_library(MUMPS_LIBRARY
    NAMES "cmumps" "dmumps" "mumps_common" "pord" "smumps" "zmumps" 
    PATHS ${MUMPS_ROOT}
    PATH_SUFFIXES "lib" "lib${_BITS}" "lib/${CMAKE_LIBRARY_ARCHITECTURE}"
	NO_DEFAULT_PATH
    )
endif()
if(NOT MUMPS_LIBRARY)
  message(STATUS "Directory with the MUMPS library not found")
  return()
endif()
list(APPEND CMAKE_REQUIRED_LIBRARIES "${MUMPS_LIBRARY}")


include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(mumps DEFAULT_MSG MUMPS_INCLUDE_DIR MUMPS_LIBRARY)
mark_as_advanced(MUMPS_INCLUDE_DIR MUMPS_LIBRARY)

# if both headers and library are found, store results
if(MUMPS_FOUND)
  set(MUMPS_INCLUDE_DIRS ${MUMPS_INCLUDE_DIR})
  set(MUMPS_LIBRARIES ${MUMPS_LIBRARY})
endif()

