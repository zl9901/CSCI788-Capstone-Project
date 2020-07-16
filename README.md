# CSCI-788 Capstone Project
> Use NLP to Answer Key Questions From Scientific Literature

[![NPM Version][npm-image]][npm-url]
[![Build Status][travis-image]][travis-url]
[![Downloads Stats][npm-downloads]][npm-url]

In December 2019, the World Health Organization identified a new coronavirus disease with severe pneumatic symptoms, which they named COVID-19. This paper uses natural language processing to answer questions from the scientific literature that can help the public better understand COVID-19. We explore both unsupervised learning (k-means) and supervised learning (random forests). Medical staff could then more efficiently search for relevant articles, using the clusters that our analysis provides. They could also categorize unsorted articles according to the schema that our models provide. Future research could further develop and confirm associations between scientific articles via sentiment analysis. 

![](header1.png)
![](header2.png)

## Installation

OS X & Linux:

```sh
npm install my-crazy-module --save
```

Windows:

```sh
edit autoexec.bat
```

## Table of Contents 
- How to read papers in the database
- Baseline of this project
- The implementation of LDA algorithm
- The implementation of GMM algorithm
- Learning curve with TF-IDF feature extraction
- Learning curve with LDA transform feature extraction
- The implementation of BERT algorithm
- How to test performance

## Usage example

This project is based on python environment, and the coding can be implemented via any python-related IDE.


## Release History

* How to read papers in the database[read_articles.py]
    * All the papers are stored in JSON format, we operate the JSON files to get the content of the paper. 
* Baseline of this project[csci788_capstone.py]
    * Data cleaning
    * TF-IDF feature extraction
    * Keywords word cloud visualization
    * Keywords histogram visualization
    * Test the performance of all 3000 papers training dataset
* The implementation of LDA algorithm[lda_implementation.py]
    * Log likelihood score visualization from grid search output
    * Create topic matrix document
    * Create topic keyword document
    * Create LDA html format visualization
    * Show top n keywords for each topic
* 0.1.0
    * The first proper release
    * CHANGE: Rename `foo()` to `bar()`
* 0.0.1
    * Work in progress

## Meta

Zhuo Liu  – zl9901@rit.edu

Rochester Institute of Technology


## Contributing

1. Fork it (<https://github.com/yourname/yourproject/fork>)
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request

<!-- Markdown link & img dfn's -->
[npm-image]: https://img.shields.io/npm/v/datadog-metrics.svg?style=flat-square
[npm-url]: https://npmjs.org/package/datadog-metrics
[npm-downloads]: https://img.shields.io/npm/dm/datadog-metrics.svg?style=flat-square
[travis-image]: https://img.shields.io/travis/dbader/node-datadog-metrics/master.svg?style=flat-square
[travis-url]: https://travis-ci.org/dbader/node-datadog-metrics
[wiki]: https://github.com/yourname/yourproject/wiki
