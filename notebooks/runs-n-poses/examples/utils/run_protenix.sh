protenix predict \
    --input "../inputs/protenix/8c3u__1__1.A__1.C.json" \
    --out_dir "../outputs/protenix" \
    --seeds "$(
        for i in {1..5}; do
        num=$(od -An -N4 -t u4 /dev/urandom | tr -d ' ')
        printf "%s" "$num"
        [ "$i" -lt 5 ] && printf ","
        done
    )"


