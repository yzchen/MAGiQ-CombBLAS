# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/cheny0l/work/db245/CombBLAS_beta_16_1

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/cheny0l/work/db245/CombBLAS_beta_16_1/build

# Include any dependencies generated for this target.
include CMakeFiles/CommGridlib.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/CommGridlib.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/CommGridlib.dir/flags.make

CMakeFiles/CommGridlib.dir/CommGrid.o: CMakeFiles/CommGridlib.dir/flags.make
CMakeFiles/CommGridlib.dir/CommGrid.o: ../CommGrid.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/cheny0l/work/db245/CombBLAS_beta_16_1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/CommGridlib.dir/CommGrid.o"
	/usr/bin/mpicxx   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/CommGridlib.dir/CommGrid.o -c /home/cheny0l/work/db245/CombBLAS_beta_16_1/CommGrid.cpp

CMakeFiles/CommGridlib.dir/CommGrid.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/CommGridlib.dir/CommGrid.i"
	/usr/bin/mpicxx  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/cheny0l/work/db245/CombBLAS_beta_16_1/CommGrid.cpp > CMakeFiles/CommGridlib.dir/CommGrid.i

CMakeFiles/CommGridlib.dir/CommGrid.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/CommGridlib.dir/CommGrid.s"
	/usr/bin/mpicxx  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/cheny0l/work/db245/CombBLAS_beta_16_1/CommGrid.cpp -o CMakeFiles/CommGridlib.dir/CommGrid.s

CMakeFiles/CommGridlib.dir/CommGrid.o.requires:

.PHONY : CMakeFiles/CommGridlib.dir/CommGrid.o.requires

CMakeFiles/CommGridlib.dir/CommGrid.o.provides: CMakeFiles/CommGridlib.dir/CommGrid.o.requires
	$(MAKE) -f CMakeFiles/CommGridlib.dir/build.make CMakeFiles/CommGridlib.dir/CommGrid.o.provides.build
.PHONY : CMakeFiles/CommGridlib.dir/CommGrid.o.provides

CMakeFiles/CommGridlib.dir/CommGrid.o.provides.build: CMakeFiles/CommGridlib.dir/CommGrid.o


# Object files for target CommGridlib
CommGridlib_OBJECTS = \
"CMakeFiles/CommGridlib.dir/CommGrid.o"

# External object files for target CommGridlib
CommGridlib_EXTERNAL_OBJECTS =

libCommGridlib.a: CMakeFiles/CommGridlib.dir/CommGrid.o
libCommGridlib.a: CMakeFiles/CommGridlib.dir/build.make
libCommGridlib.a: CMakeFiles/CommGridlib.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/cheny0l/work/db245/CombBLAS_beta_16_1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libCommGridlib.a"
	$(CMAKE_COMMAND) -P CMakeFiles/CommGridlib.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/CommGridlib.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/CommGridlib.dir/build: libCommGridlib.a

.PHONY : CMakeFiles/CommGridlib.dir/build

CMakeFiles/CommGridlib.dir/requires: CMakeFiles/CommGridlib.dir/CommGrid.o.requires

.PHONY : CMakeFiles/CommGridlib.dir/requires

CMakeFiles/CommGridlib.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/CommGridlib.dir/cmake_clean.cmake
.PHONY : CMakeFiles/CommGridlib.dir/clean

CMakeFiles/CommGridlib.dir/depend:
	cd /home/cheny0l/work/db245/CombBLAS_beta_16_1/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/cheny0l/work/db245/CombBLAS_beta_16_1 /home/cheny0l/work/db245/CombBLAS_beta_16_1 /home/cheny0l/work/db245/CombBLAS_beta_16_1/build /home/cheny0l/work/db245/CombBLAS_beta_16_1/build /home/cheny0l/work/db245/CombBLAS_beta_16_1/build/CMakeFiles/CommGridlib.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/CommGridlib.dir/depend

