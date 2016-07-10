Rodgers2014
===========

This repository contains all of the code necessary to recapitulate the analyses presented in Rodgers and DeWeese (2014), Neuron.


Format
------
This repository consists of:

* Numbered subdirectories. Each one contains a series of scripts of the form main1.py, main2.py, ..., relating to a specific type of analysis. (For instance, subdirectory 2-1-behavres contains all the scripts for analyzing the behavioral results.)
* A subdirectory called 'modules'. This contains copies of my other github repositories that you need to run the code. I copied the modules to make it easier to install, and also because I removed some (but not all) of the code from the modules that is not related to the analyses presented here.

The 'modules' directory needs to be on your PYTHONPATH so that the scripts can import the modules it contains. For instance, you should be able to 'import ns5_process' and get the module in modules/ns5_process.

The scripts should be run in order: from lowest to highest numbered subdirectory, and within each subdirectory from lowest to highest numbered script. This is because some scripts depend on the output of previous scripts.


Data
----
The data is available at the CRCNS website http://crcns.org/data-sets/pfc/pfc-1/about-pfc-1 .
You should first clone this repository. Then download the file `data.zip` from CRCNS, and unzip it into the `Rodgers2014` directory that was created
when you cloned the repository. Thus, you will have directories like Rodgers2014/data/CR12B, Rodgers2014/data/CR20B, etc.

I wrote a FAQ on how to read the data on the CRCNS website. However, each script here should load the data by itself without any intervention, as long as the directory structure is the way I described above.

Important: the spike data includes both well-sorted units and bad clusters (artefacts or multi-unit). We only analyzed well-sorted units in the paper and I recommend that you do the same. This csv file (https://github.com/cxrodgers/Rodgers2014/blob/master/metadata/unit_db.csv) tells you which units are good. Only the units that are listed in that table and for which the "include" column is true are well-sorted units.

Required dependencies
---------------------
I ran all of the code with the following Python modules installed. Other versions will probably work. (Exception: older versions of sklearn had a bug in the `class_weights` parameter to LogisticRegression, so some of the analyses of evoked responses will fail if you do not use the version listed here.)

* pandas 0.10.0
* numpy 1.6.1
* scipy 0.11.0
* matplotlib 1.2.0
* statsmodels 0.5.0.dev-1bbd4ca
* rpy2 2.2.5
* sklearn 0.12.1

I used Python 2.7.3 .

You will also need the statistical language R, version 2.14.1 . (Not a Python module.)

I ran all of my analyses within ipython, version 0.12.1 .


Metadata
--------
The scripts can produce figures (always ending in *.svg or *.pdf) and/or statistics, which are written out as text files begininning with 'stat__'. I have also provided a spreadsheet at `metadata/figures_generated_by_each_script.csv` which links each figure panel and statistic with the script that produces it. So if you want to regenerate a particular figure, use this spreadsheet to find the associated script. This is subject to the usual caveat that the scripts are designed to be run in order and may assume the results of earlier scripts.
