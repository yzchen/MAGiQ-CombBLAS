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
include ReleaseTests/CMakeFiles/ParIOTest.dir/depend.make

# Include the progress variables for this target.
include ReleaseTests/CMakeFiles/ParIOTest.dir/progress.make

# Include the compile flags for this target's objects.
include ReleaseTests/CMakeFiles/ParIOTest.dir/flags.make

ReleaseTests/CMakeFiles/ParIOTest.dir/ParIOTest.o: ReleaseTests/CMakeFiles/ParIOTest.dir/flags.make
ReleaseTests/CMakeFiles/ParIOTest.dir/ParIOTest.o: ../ReleaseTests/ParIOTest.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/cheny0l/work/db245/CombBLAS_beta_16_1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object ReleaseTests/CMakeFiles/ParIOTest.dir/ParIOTest.o"
	cd /home/cheny0l/work/db245/CombBLAS_beta_16_1/build/ReleaseTests && /usr/bin/mpicxx   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/ParIOTest.dir/ParIOTest.o -c /home/cheny0l/work/db245/CombBLAS_beta_16_1/ReleaseTests/ParIOTest.cpp

ReleaseTests/CMakeFiles/ParIOTest.dir/ParIOTest.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ParIOTest.dir/ParIOTest.i"
	cd /home/cheny0l/work/db245/CombBLAS_beta_16_1/build/ReleaseTests && /usr/bin/mpicxx  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/cheny0l/work/db245/CombBLAS_beta_16_1/ReleaseTests/ParIOTest.cpp > CMakeFiles/ParIOTest.dir/ParIOTest.i

ReleaseTests/CMakeFiles/ParIOTest.dir/ParIOTest.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ParIOTest.dir/ParIOTest.s"
	cd /home/cheny0l/work/db245/CombBLAS_beta_16_1/build/ReleaseTests && /usr/bin/mpicxx  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/cheny0l/work/db245/CombBLAS_beta_16_1/ReleaseTests/ParIOTest.cpp -o CMakeFiles/ParIOTest.dir/ParIOTest.s

ReleaseTests/CMakeFiles/ParIOTest.dir/ParIOTest.o.requires:

.PHONY : ReleaseTests/CMakeFiles/ParIOTest.dir/ParIOTest.o.requires

ReleaseTests/CMakeFiles/ParIOTest.dir/ParIOTest.o.provides: ReleaseTests/CMakeFiles/ParIOTest.dir/ParIOTest.o.requires
	$(MAKE) -f ReleaseTests/CMakeFiles/ParIOTest.dir/build.make ReleaseTests/CMakeFiles/ParIOTest.dir/ParIOTest.o.provides.build
.PHONY : ReleaseTests/CMakeFiles/ParIOTest.dir/ParIOTest.o.provides

ReleaseTests/CMakeFiles/ParIOTest.dir/ParIOTest.o.provides.build: ReleaseTests/CMakeFiles/ParIOTest.dir/ParIOTest.o


# Object files for target ParIOTest
ParIOTest_OBJECTS = \
"CMakeFiles/ParIOTest.dir/ParIOTest.o"

# External object files for target ParIOTest
ParIOTest_EXTERNAL_OBJECTS =

ReleaseTests/ParIOTest: ReleaseTests/CMakeFiles/ParIOTest.dir/ParIOTest.o
ReleaseTests/ParIOTest: ReleaseTests/CMakeFiles/ParIOTest.dir/build.make
ReleaseTests/ParIOTest: libCommGridlib.a
ReleaseTests/ParIOTest: libMPITypelib.a
ReleaseTests/ParIOTest: libMemoryPoollib.a
ReleaseTests/ParIOTest: libHashlib.a
ReleaseTests/ParIOTest: libmmiolib.a
ReleaseTests/ParIOTest: ReleaseTests/CMakeFiles/ParIOTest.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/cheny0l/work/db245/CombBLAS_beta_16_1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ParIOTest"
	cd /home/cheny0l/work/db245/CombBLAS_beta_16_1/build/ReleaseTests && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ParIOTest.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
ReleaseTests/CMakeFiles/ParIOTest.dir/build: ReleaseTests/ParIOTest

.PHONY : ReleaseTests/CMakeFiles/ParIOTest.dir/build

ReleaseTests/CMakeFiles/ParIOTest.dir/requires: ReleaseTests/CMakeFiles/ParIOTest.dir/ParIOTest.o.requires

.PHONY : ReleaseTests/CMakeFiles/ParIOTest.dir/requires

ReleaseTests/CMakeFiles/ParIOTest.dir/clean:
	cd /home/cheny0l/work/db245/CombBLAS_beta_16_1/build/ReleaseTests && $(CMAKE_COMMAND) -P CMakeFiles/ParIOTest.dir/cmake_clean.cmake
.PHONY : ReleaseTests/CMakeFiles/ParIOTest.dir/clean

ReleaseTests/CMakeFiles/ParIOTest.dir/depend:
	cd /home/cheny0l/work/db245/CombBLAS_beta_16_1/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/cheny0l/work/db245/CombBLAS_beta_16_1 /home/cheny0l/work/db245/CombBLAS_beta_16_1/ReleaseTests /home/cheny0l/work/db245/CombBLAS_beta_16_1/build /home/cheny0l/work/db245/CombBLAS_beta_16_1/build/ReleaseTests /home/cheny0l/work/db245/CombBLAS_beta_16_1/build/ReleaseTests/CMakeFiles/ParIOTest.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : ReleaseTests/CMakeFiles/ParIOTest.dir/depend

