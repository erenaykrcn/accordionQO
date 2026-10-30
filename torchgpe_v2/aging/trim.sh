for f in results/*.hdf5; do
    base=$(basename "$f")
    t="results_trimmed/$base"

    if [ -f "$t" ]; then
        orig=$(stat -f "%z" "$f")
        trim=$(stat -f "%z" "$t")

        if [ "$trim" -lt "$orig" ]; then
            echo "DELETE: $f"
            rm "$f"
        else
            echo "KEEP:   $f  (trimmed file is not smaller)"
        fi
    else
        echo "KEEP:   $f  (NO TRIMMED COPY)"
    fi
done
