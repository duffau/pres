#!/bin/bash
set -e
rm -rf myrepo

# Command to live monitor .git folder: watch -n 0.2 tree -a .
# start snippet make-myrepo
mkdir myrepo
cd myrepo
# end snippet make-myrepo

# Making a git repo the regular way
# start snippet git-init
git init
# end snippet git-init
# start snippet git-rm
rm -rf .git
# end snippet git-rm

# start snippet git-init-manual
# Inititalizing repo manually
mkdir .git
mkdir .git/objects
mkdir .git/refs
echo "ref: refs/heads/master" > .git/HEAD
# end snippet git-init-manual

# start snippet git-status
git status
# end snippet git-status

# Making som file
echo "Some content" > first_file.txt

# The git datamodel basic
# -----------------------
# git stores three types of objects:
# - blob
# - tree
# - commit

# All these objects are saved in .git/objects and are named as the SHA1 hash of
# their content.

# In the refs-folder are named references to commits. refs/heads are "branches"
# which is just a named reference to a specific commit hash.


# Adding a file to git manually
# +++++++++++++++++++++++++++++

# Ading a file using the git plumbing commands
# --------------------------------------------

# First we make a "blob" object input in the git database

# All files in git are refered to by their hash
cat first_file.txt | git hash-object --stdin

# To add the file to the blob database we add the "-w" flag for write
cat first_file.txt | git hash-object --stdin -w
# The blob is created and appears in the object-folder: 0e/e389557af36e8d030f7fdc724f2185280c4dd4 

# A file blob is created, which is the compressed content of the file.
git cat-file -t 0ee38   # the object type
# > blob
git cat-file -p 0ee38   # the object content (decompressed and pretty printed)
# > Some content
git cat-file -s 0ee38   # The size of the file (decompressed)
# > 13

# git does not store changes. Everytime a "git add" happens a new blob is created
# which is an absolute image of the content of the file.

# Re-creating thre blob object manually
# -------------------------------------
# That was to easy let's start over making a blob object
rm -rf .git/objects/0e

# The hash is calculated by
# echo "blob <size of file in digits>\0<file content>" | sha1sum
content=$(cat first_file.txt)
size=$((${#content}+1))
header="blob $size"
obj="$header\0$content"
echo -e $obj | sha1sum
# > 0ee389557af36e8d030f7fdc724f2185280c4dd4

# Making the objects folder
mkdir .git/objects/0e

# Now we need to make the input to the file
# First the file path
objpath=.git/objects/0e/$(echo -e $obj | sha1sum | cut -c3-40)

# then the content of the blob object
# we take the object <header>\0<content> an compress it with zlib level 1 (pigz -z1)
echo -e $obj
echo -e $obj | pigz -zf1 > $objpath

git cat-file -t 0ee38 
git cat-file -s 0ee38
git cat-file -p 0ee38 

# Hurray! Now we have added a blob object to the git object database!

# Staging the blob
# ----------------
git status
# > On branch master
# >
# > No commits yet
# >
# > Untracked files:
# >   (use "git add <file>..." to include in what will be committed)
# >
# > 	first_file.txt
# >
# > nothing added to commit but untracked files present (use "git add" to track)

# The blob is not staged yet. We need to add the blob to the staging area, also called "the index".

# source ./git-utils.sh


# Adding blob to the staging area/index
git update-index --add first_file.txt 
# 100644 is unix file permissions

	# /////////////////////////
	# Unix permission mode bits
	# /////////////////////////
	# The unix permissions are in octal (0-7) with the structure
	#     Type|---|Perm bits
	# bin 1000 000 110100100
	# oct 1 0   0   6  4  4
	# where the type is either
	# 1000 -> regular file
	# 1010 -> sym link
	# 1110 -> gitlink
	# and the permission bits are grouped in three bits for each group (user, 
	# group, others). 
	printf "%o\n" 0x$(stat -c '%f' first_file.txt)


# This creates the index-file (if not allready present) 
# and adds the blob-hash, the file mode and filepath to the index (also known as staging area).
# Adding a file to the staging area amounts to taking a snapshot of the content,
# the filepath and the file mode.

git ls-files --stage
# This command shows the files in the staging area (index) 

# We have created file, a corresponding blob, put a reference to the 
# blob in the index/staging. But we are not ready to create a commit-object yet.
# We need a tree-object.

# Creating a tree object
# ----------------------

# The tree represents a UNIX-like directory entry. 
# It stores the path of all files in a directory, 
# the filemodes and a reference to the corresponding blobs,
# as well as references to other sub-trees.

# git cat-file -p HEAD^{tree}
# > fatal: Not a valid object name HEAD^{tree}
# This command tries to print the content of the tree of HEAD
# but no such object exists so it fails.

# File content -> blob
# file mode, blob hash, filepath -> tree (represents a directory)
git write-tree

# This creates a new file in "objects/2d/40b0... 

# The newly created object is of type tree
git cat-file -t 2d40
# > tree

git cat-file -p 2d40
# > 100644 blob 0ee389557af36e8d030f7fdc724f2185280c4dd4	first_file.txt


# Creating a commit object
# ------------------------

# The third and final object type in the git datamodel is the commit-object.
# The commit is a reference to the tree, which in turn refers to the blob and the file in
# the repo's folder.

# The structure of the commit is at text file:

# commit {size of content}\0{content}

COMMIT_SHA1=$(git commit-tree 2d40b0701c6d0556d6d69fc400dd40f1171a9bf1 -m "Initial commit")
echo $COMMIT_SHA1
# > ab4131d5e0740372e18ee84279f82fa09b05aceb

# The hash displayed is the commit hash. We see a third object in the objects-folder, named 
# with the SHA1 hash. 

git status
# > On branch master
# >
# > No commits yet
# >
# > Changes to be committed:
# >  (use "git rm --cached <file>..." to unstage)
# >
# >	new file:   first_file.txt

# Git status is not happy yet since theres are no refs. Thats is, named references to the commit.
cat .git/HEAD
# > ref: refs/heads/master
# Is a symbolic ref to master

	# ///////////////////
	# Detached head state
	# ///////////////////
	# If the HEAD is somethig like
	# ref: ab4131d5e0740372e18ee84279f82fa09b05aceb
	# This is known as "detached head state".
	# The reference is to a specific commit hash

# There is no refs/head/master so "git status" which reads the HEAD file does not see the commit.
# To create the reference to the latest commmit we write the commit hash to the refs/head/master file.
mkdir .git/refs/heads
echo $COMMIT_SHA1 > .git/refs/heads/master

git status
# > On branch master
# > nothing to commit, working tree clean

# Hurray! Git status is happy. We have performed a valid git commit,
# and we now have a minimal valid git repo with 
# - a single blob object which is the content of the commited file
# - a tree object representing the directory of the file (file mode, path and ref to blob)
# - a commit object referencing the tree
# - a named reference to the commit object in the master "branch" aka. the refs/heads/master file 

exit 0

# Git branches/refs
# -----------------
git branch new_feature

# A new file in .git/refs/heads is created. This file contains the SHA1 of the commit.
cat .git/refs/heads/new_feature

# When you checkout a branch the HEAD file cheanges
git checkout new_feature
cat .git/HEAD
# > ref: refs/heads/new_feature

git checkout master

# Tags are also refs
git tag v0.1.0
cat .git/refs/tags/v0.1.0

# Git stores a full snapshot - untill it doesn't
# ----------------------------------------------
echo "Some other content" >> first_file.txt

git add first_file.txt

# Looking at the ojects we see the new blob (3a96...) which is a copy of first_file with the new content.
# The previous state of the file is store in blob 0ee38
git cat-file -t 3a96
git cat-file -p 0ee3
git cat-file -p 3a96

# The form of storing all objects in the objects folder, with a blob for each version is called "loose" objects.
# Eventhough these files are compressed with zlib, it would still take space the space of the fulle for each addition.

git commit first_file.txt -m "More content"

# Packfiles
# --------- 

# Generating 40K of random digits (0-9)
RANDOM=42;for i in {1..40000};do;printf $(($RANDOM % 10)) >> second_file.txt;done;

cat second_file.txt | git hash-object --stdin 
# > c9e51f...

git add second_file.txt
git commit -m "Big A file"

git cat-file -s c9e5
# > 40000

# Altering file just little bit, which of course changes the SHA1 hash 
echo b >> second_file.txt

# Adding will create a new blob
git add second_file.txt
# with hash 283ac0...

# The new blob has only two extra bytes, but takes up the an additional 40k of space.
git cat-file -s 283a

# The file is compressed with zlib but it still take up 200 or so bytes. This file is "easy" to 
# compress since it's just a repetition of a sinlge character. In general each addition would take
# save a (compresed) copy of the whole file, which would grow rapidely. 


# To handle this problem git "packs" the files and stores delta og file with similar names and sizes.
# To invoke the packaging of loose file use git gc (garbage collection).
# This is done automatocally when there many loose files or when pushing to a remote

git gc

# All objects are now packed into the "pack" folder, with a index and a binary file. The index
# containes offsets for each origin object.

echo "SHA-1                                    type bytes bytes-pack offset depth base-SHA-1"

git verify-pack --verbose .git/objects/pack/pack-*.pack

# SHA-1     type   size  size-in-packfile offset-in-packfile depth base-SHA-1
# 283ac0... blob   40002 68               460
# ...
# c9e51f... blob   9     20               613                 1    283ac0e04b...

# Wee see that the most recent blob (283a) takes up 40K of space unpacked but only 68 bytes in the pack. 
# Thats because it's esy to compress.
# The previous version of the second_file.txt (c9e5) only take up 9 bytes uncompressed. That's a testemony
# of git only tsoring the delta between the two files in the pack.




