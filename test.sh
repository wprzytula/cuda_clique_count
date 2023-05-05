cd tests/poprawnosciowe_dominik
for test in `ls input`; do 
    ../../kcliques input/$test 12 tmp.file
    cmp tmp.file output/$test
done
