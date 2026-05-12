# Library Import Workflow

This folder contains a repeatable workflow for cleaning the library workbook, splitting the data into two tables, importing into MySQL, and finding common books.

## Files

- Source workbook with headers: `outputs/Combined library with headers.xlsx`
- Fixed CSV for books with accession number: `outputs/books_with_accession_no_fixed.csv`
- Fixed CSV for books without accession number: `outputs/books_without_accession_no_fixed.csv`

## 1. Understand the workbook

The worksheet contains book records. The important columns are:

1. `Accession No.`
2. `Book Title`
3. `Author`
4. `Price (INR)`
5. `Price ($)`
6. `Price (Pound)`
7. `Publisher`
8. `Issue / Lost / Available`
9. `Book No. / Barcode`
10. `Book Shelf No.`
11. `No. of Copies`
12. `S. No.`

Some rows have a duplicated accession number in column 2. In those rows, the real book title starts in column 3. That is why the export needs a small cleanup step.

## 2. Split the workbook into two groups

Use this rule:

- If `Accession No.` is present, put the row in `books_with_accession_no`
- If `Accession No.` is blank, put the row in `books_without_accession_no`

Also fix the shifted rows:

- If column 1 and column 2 contain the same accession number, drop the duplicate value from column 2
- Move the book title back into the `Book Title` column

## 3. Export the two CSV files

The cleaned files are:

- `outputs/books_with_accession_no_fixed.csv`
- `outputs/books_without_accession_no_fixed.csv`

If you need to regenerate them, the logic is:

1. Read the workbook.
2. Check each row.
3. If accession number exists, write to the "with accession" CSV.
4. If accession number is blank, write to the "without accession" CSV.
5. If a row has the repeated accession number pattern in column 2, remove the duplicate and keep the title in column 3.

## 4. Create the MySQL tables

Run this in MySQL:

```sql
CREATE TABLE books_with_accession_no (
  accession_no INT NOT NULL,
  book_title VARCHAR(255),
  author VARCHAR(255),
  price_inr DECIMAL(10,2),
  price_usd DECIMAL(10,2),
  price_pound DECIMAL(10,2),
  publisher VARCHAR(255),
  issue_lost_available VARCHAR(100),
  book_no_barcode VARCHAR(50),
  book_shelf_no VARCHAR(50),
  no_of_copies INT,
  s_no INT
);

CREATE TABLE books_without_accession_no (
  accession_no INT NULL,
  book_title VARCHAR(255),
  author VARCHAR(255),
  price_inr DECIMAL(10,2),
  price_usd DECIMAL(10,2),
  price_pound DECIMAL(10,2),
  publisher VARCHAR(255),
  issue_lost_available VARCHAR(100),
  book_no_barcode VARCHAR(50),
  book_shelf_no VARCHAR(50),
  no_of_copies INT,
  s_no INT
);
```

## 5. Enable local file loading

`LOAD DATA LOCAL INFILE` only works if local loading is enabled on both the MySQL client and server.

Check the setting:

```sql
SHOW VARIABLES LIKE 'local_infile';
```

If needed, enable it:

```sql
SET GLOBAL local_infile = 1;
```

Reconnect the client with:

```bash
mysql --local-infile=1 -u your_user -p your_database
```

## 6. Import the CSV files

Always specify the column list so MySQL maps fields correctly:

```sql
TRUNCATE TABLE books_with_accession_no;
TRUNCATE TABLE books_without_accession_no;

LOAD DATA LOCAL INFILE '/Users/aruneshsingh/Documents/Codex/2026-05-12/files-mentioned-by-the-user-combined/outputs/books_with_accession_no_fixed.csv'
INTO TABLE books_with_accession_no
FIELDS TERMINATED BY ','
OPTIONALLY ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 LINES
(accession_no, book_title, author, price_inr, price_usd, price_pound, publisher, issue_lost_available, book_no_barcode, book_shelf_no, no_of_copies, s_no);

LOAD DATA LOCAL INFILE '/Users/aruneshsingh/Documents/Codex/2026-05-12/files-mentioned-by-the-user-combined/outputs/books_without_accession_no_fixed.csv'
INTO TABLE books_without_accession_no
FIELDS TERMINATED BY ','
OPTIONALLY ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 LINES
(accession_no, book_title, author, price_inr, price_usd, price_pound, publisher, issue_lost_available, book_no_barcode, book_shelf_no, no_of_copies, s_no);
```

## 7. Verify the import

Check row counts:

```sql
SELECT COUNT(*) FROM books_with_accession_no;
SELECT COUNT(*) FROM books_without_accession_no;
```

Spot check the data:

```sql
SELECT * FROM books_with_accession_no LIMIT 5;
SELECT * FROM books_without_accession_no LIMIT 5;
```

## 8. Find common books in both tables

Match by title and author:

```sql
SELECT DISTINCT
  w.book_title,
  w.author
FROM books_with_accession_no AS w
INNER JOIN books_without_accession_no AS n
  ON TRIM(LOWER(w.book_title)) = TRIM(LOWER(n.book_title))
 AND TRIM(LOWER(w.author)) = TRIM(LOWER(n.author));
```

If you want to match more strictly, include publisher too:

```sql
SELECT DISTINCT
  w.book_title,
  w.author,
  w.publisher
FROM books_with_accession_no AS w
INNER JOIN books_without_accession_no AS n
  ON TRIM(LOWER(w.book_title)) = TRIM(LOWER(n.book_title))
 AND TRIM(LOWER(w.author)) = TRIM(LOWER(n.author))
 AND TRIM(LOWER(COALESCE(w.publisher, ''))) = TRIM(LOWER(COALESCE(n.publisher, '')));
```

## 9. Notes on the anomaly

Rows around the later part of the sheet contain a duplicate accession number in column 2. If you see output like:

- column 1 = accession number
- column 2 = same accession number again
- column 3 = book title

then the row must be corrected before importing.

## 10. Recommended workflow next time

1. Start from the workbook.
2. Add or verify the header row.
3. Split rows by accession number.
4. Fix rows where column 2 repeats the accession number.
5. Export two CSV files.
6. Truncate both MySQL tables.
7. Load both CSVs using explicit column names.
8. Run the common-book query.

