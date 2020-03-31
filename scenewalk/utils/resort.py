#!/usr/bin/env python3
"""
Little commandline tool to sort estimation files into the folders.

Go into the folder where all the estimation files are and run
´´´
python3 -m scenewalk.utils.resort "2019" 5 -d
´´´
"""

import glob
import shutil
import re
from argparse import ArgumentParser

def main():
    """
    Parse Commandline Arguments
    """
    parser = ArgumentParser()
    parser.add_argument("id", help="estimation id",
                        type=str)
    parser.add_argument("max_vp", help="highest vp number",
                        type=int)
    parser.add_argument("-d",
                        "--dry_run",
                        help="just print upload commands",
                        action="store_true")
    parser.add_argument("-o",
                        "--orson", help="orson cluster log files",
                        action="store_true")
    args = parser.parse_args()
    move = not args.dry_run
    orson = args.orson
    if orson:
        move_files_orson(args.id, args.max_vp, move)
    else:
        move_files(args.id, args.max_vp, move)

    print("Done!")

def move_files(estim_id, nvp, move):
    """
    moving utility for estimations run on ex_bio_psy server
    """
    estim_folder_list = glob.glob("estim_"+estim_id+"*[0-9]")
    estim_folder_list_sorted = [None for el in range(nvp)]
    i = 0
    for p in estim_folder_list:
        #print(p)
        i += 1
        mat = re.search(r"vp([0-9]*)", p)
        vpnr = int(mat.group(1))
        #print(vpnr)
        if estim_folder_list_sorted[vpnr] is not None:
            print("You seem to have 2 estims for the same subject")
        estim_folder_list_sorted[vpnr] = p
    #print(estim_folder_list)
    logs_counter = 0
    for i, est in enumerate(estim_folder_list_sorted):
        logs_counter = i+1
        if est is None:
            print("There is no folder for subject"+ str(i))
            continue
        # MOVE ERRLOGS
        errlog_l = glob.glob("*[[]"+str(logs_counter)+"[]].err.log")
        if not errlog_l:
            print("There is no .err.log file for for subject"+ str(i))
        else:
            errlog = errlog_l[0]
            if move:
                shutil.move(errlog, est)
            else:
                print("move "+ errlog + " to " + est)

        # MOVE LOGS
        log_l = glob.glob("*[[]"+str(logs_counter)+"[]].log")
        if not log_l:
            print("There is no .log file for for subject"+ str(i))
        else:
            log = log_l[0]
            if move:
                shutil.move(log, est)
            else:
                print("move "+ log + " to " + est)

        # MOVE RESULTS
        resfiles = glob.glob("estim_*vp"+str(logs_counter-1)+"_*")
        for rf in resfiles:
            if move:
                shutil.move(rf, est)
            else:
                print("move " + rf + " to " + est)

def move_files_orson(estim_id, nvp, move):
    """
    moving utility for estimations run on orson server
    """
    print("orson")
    estim_folder_list = glob.glob("estim_"+estim_id+"*[0-9]")
    estim_folder_list_sorted = [None for el in range(nvp)]
    i = 0
    for p in estim_folder_list:
        #print(p)
        i += 1
        mat = re.search(r"vp([0-9]*)", p)
        vpnr = int(mat.group(1))
        #print(vpnr)
        if estim_folder_list_sorted[vpnr] is not None:
            print("You seem to have 2 estims for the same subject")
        estim_folder_list_sorted[vpnr] = p

    print("liiist", estim_folder_list_sorted)
    logs_l = glob.glob("est_slurm*.out")
    print(logs_l)

    for l in logs_l:
        if not l:
            print("There is no .log file for for subject")
        else:
            with open(l) as f:
                first_line = f.readline()
                vpnr = int(first_line[4:])
            print(vpnr)

            logpath = l
            errlogpath = l[:-4]+".err"
            est = estim_folder_list_sorted[vpnr-1]
            if move:
                shutil.move(logpath, est)
                shutil.move(errlogpath, est)
            else:
                print("move " + logpath + " to " + est)
                print("move " + errlogpath + " to " + est)

            resfiles = glob.glob("estim_*vp"+str(vpnr-1)+"_*")
            for rf in resfiles:
                if move:
                    shutil.move(rf, est)
                else:
                    print("move " + rf + " to " + est)

if __name__ == "__main__":
    print("main")
    main()
