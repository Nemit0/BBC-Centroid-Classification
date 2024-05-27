file_list = $(ls)

function endswith(str, suffix) {
    return str[-#suffix] == suffix
}
for file in $file_list; do
    if endswith($file, ".py") {
        cat $file
    }
done

# remove function endswith
remove endswith