cd tests/poprawnosciowe_dominik
for test in `ls input`; do
    echo -e "\n	Running test ${test}..."
    ../../kcliques input/$test 12 tmp.file
    diff tmp.file output/$test
done
