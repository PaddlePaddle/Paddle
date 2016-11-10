
CREATE TABLE download_info (
    id integer NOT NULL,
    date date,
    key character varying,
    value character varying
);



CREATE TABLE usage_info (
    id integer NOT NULL,
    ip inet,
    type integer,
    "time" timestamp without time zone,
    payload json,
    create_time timestamp without time zone
);

