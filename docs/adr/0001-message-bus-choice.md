# ADR 0001: Redis Streams as the Runtime Message Bus

## Status

Accepted

## Context

The system needs a simple, observable bus for vehicle counts, phase commands,
phase states, and reasoning logs. The portfolio-scale target is a single
intersection or a small set of intersections, not a fleet-wide deployment.

## Decision

Use Redis Streams for the current implementation.

## Consequences

Redis keeps local development and Docker Compose lightweight while still
demonstrating durable stream semantics and decoupled services. Kafka remains a
reasonable future option if the system grows to many intersections, high
retention needs, or cross-team stream processing.
