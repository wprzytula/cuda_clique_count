#!/bin/zsh

declare -A k
k[poprawnosciowe_dominik]=12
k[bartka_tests]=11

cd tests
for testdir in `ls`; do
    [ -d $testdir ] && for testname in $(ls "${testdir}/input"); do
        echo -e "\n	Running test ${testname}..."
        ../kcliques ${testdir}/input/$testname $k[${testdir}] tmp.file
        diff tmp.file ${testdir}/output/$testname || exit 1
    done
done
