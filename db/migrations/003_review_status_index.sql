-- depends: 002_review_index
CREATE INDEX idx_applications_review_status
    ON applications_database (review_status);
