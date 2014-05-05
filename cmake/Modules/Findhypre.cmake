# - Try to find Eigen3 lib
#
# This module supports requiring a minimum version, e.g. you can do
#   find_package(Eigen3 3.1.2)
# to require version 3.1.2 or newer of Eigen3.
#
# Once done this will define
#
#  HYPRE_FOUND - system has hypre lib with correct version
#  HYPRE_INCLUDE_DIRS - the hypre include directory
#  HYPRE_LIBRARIES   - the hypre library.

# Copyright (c) 2006, 2007 Montel Laurent, <montel@kde.org>
# Copyright (c) 2008, 2009 Gael Guennebaud, <g.gael@free.fr>
# Copyright (c) 2009 Benoit Jacob <jacob.benoit.1@gmail.com>
# Redistribution and use is allowed according to the terms of the 2-clause BSD license.

# find out the size of a pointer. this is required to only search for
# libraries in the directories relevant for the architecture
if (CMAKE_SIZEOF_VOID_P)
  math (EXPR _BITS "8 * ${CMAKE_SIZEOF_VOID_P}")
endif (CMAKE_SIZEOF_VOID_P)

if (hypre_ROOT)
 set (HYPRE_ROOT "${hypre_ROOT}")
endif (hypre_ROOT)

if (NOT HYPRE_INCLUDE_DIR)
  if (HYPRE_ROOT)
	find_path (HYPRE_INCLUDE_DIR
	  NAMES "HYPRE.h"
	  PATHS ${HYPRE_ROOT}
	  PATH_SUFFIXES "hypre-2.6.0b" "include"
	  NO_DEFAULT_PATH
	  )
  endif (HYPRE_ROOT)
endif (NOT HYPRE_INCLUDE_DIR)

if(NOT HYPRE_INCLUDE_DIR)
  message(STATUS "Directory with the HYPRE include not found")
  return()
endif() 

# look for actual hypre library
if (NOT HYPRE_LIBRARY)
  find_library(HYPRE_LIBRARY
    NAMES "HYPRE" 
    PATHS ${HYPRE_ROOT}
    PATH_SUFFIXES "lib" "lib${_BITS}" "lib/${CMAKE_LIBRARY_ARCHITECTURE}"
	NO_DEFAULT_PATH
    )
endif()
if(NOT HYPRE_LIBRARY)
  message(STATUS "Directory with the HYPRE library not found")
  return()
endif()
list(APPEND CMAKE_REQUIRED_LIBRARIES "${HYPRE_LIBRARY}")


include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(hypre DEFAULT_MSG HYPRE_INCLUDE_DIR HYPRE_LIBRARY)
mark_as_advanced(HYPRE_INCLUDE_DIR HYPRE_LIBRARY)

# if both headers and library are found, store results
if(HYPRE_FOUND)
  set(HYPRE_INCLUDE_DIRS ${HYPRE_INCLUDE_DIR})
  set(HYPRE_LIBRARIES ${HYPRE_LIBRARY})
endif()

