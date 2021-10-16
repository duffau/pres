function gitobj() {
	local filename=$1
	local type=$2
	local content=$(cat $filename)
	local size=$((${#content}+1))
	local header="blob $size"
	echo -e "$header\0$content"
}


function gitsha() {        
	local content=$(gitobj $1 $2)
	echo -e $content | sha1sum | cut -c1-40
}

function gitcomp() {        
	local content=$(gitobj $1 $2)
	echo -e $content | pigz -zf1
}


