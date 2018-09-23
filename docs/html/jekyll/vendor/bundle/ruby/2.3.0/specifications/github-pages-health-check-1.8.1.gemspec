# -*- encoding: utf-8 -*-
# stub: github-pages-health-check 1.8.1 ruby lib

Gem::Specification.new do |s|
  s.name = "github-pages-health-check"
  s.version = "1.8.1"

  s.required_rubygems_version = Gem::Requirement.new(">= 0") if s.respond_to? :required_rubygems_version=
  s.require_paths = ["lib"]
  s.authors = ["GitHub, Inc."]
  s.date = "2018-05-07"
  s.description = "Checks your GitHub Pages site for commons DNS configuration issues."
  s.email = "support@github.com"
  s.homepage = "https://github.com/github/github-pages-health-check"
  s.licenses = ["MIT"]
  s.required_ruby_version = Gem::Requirement.new(">= 2.2.0")
  s.rubygems_version = "2.5.2.1"
  s.summary = "Checks your GitHub Pages site for commons DNS configuration issues"

  s.installed_by_version = "2.5.2.1" if s.respond_to? :installed_by_version

  if s.respond_to? :specification_version then
    s.specification_version = 4

    if Gem::Version.new(Gem::VERSION) >= Gem::Version.new('1.2.0') then
      s.add_runtime_dependency(%q<addressable>, ["~> 2.3"])
      s.add_runtime_dependency(%q<dnsruby>, ["~> 1.60"])
      s.add_runtime_dependency(%q<octokit>, ["~> 4.0"])
      s.add_runtime_dependency(%q<public_suffix>, ["~> 2.0"])
      s.add_runtime_dependency(%q<typhoeus>, ["~> 1.3"])
      s.add_development_dependency(%q<dotenv>, ["~> 1.0"])
      s.add_development_dependency(%q<gem-release>, ["~> 0.7"])
      s.add_development_dependency(%q<pry>, ["~> 0.10"])
      s.add_development_dependency(%q<rspec>, ["~> 3.0"])
      s.add_development_dependency(%q<rubocop>, ["~> 0.52"])
      s.add_development_dependency(%q<webmock>, ["~> 1.21"])
    else
      s.add_dependency(%q<addressable>, ["~> 2.3"])
      s.add_dependency(%q<dnsruby>, ["~> 1.60"])
      s.add_dependency(%q<octokit>, ["~> 4.0"])
      s.add_dependency(%q<public_suffix>, ["~> 2.0"])
      s.add_dependency(%q<typhoeus>, ["~> 1.3"])
      s.add_dependency(%q<dotenv>, ["~> 1.0"])
      s.add_dependency(%q<gem-release>, ["~> 0.7"])
      s.add_dependency(%q<pry>, ["~> 0.10"])
      s.add_dependency(%q<rspec>, ["~> 3.0"])
      s.add_dependency(%q<rubocop>, ["~> 0.52"])
      s.add_dependency(%q<webmock>, ["~> 1.21"])
    end
  else
    s.add_dependency(%q<addressable>, ["~> 2.3"])
    s.add_dependency(%q<dnsruby>, ["~> 1.60"])
    s.add_dependency(%q<octokit>, ["~> 4.0"])
    s.add_dependency(%q<public_suffix>, ["~> 2.0"])
    s.add_dependency(%q<typhoeus>, ["~> 1.3"])
    s.add_dependency(%q<dotenv>, ["~> 1.0"])
    s.add_dependency(%q<gem-release>, ["~> 0.7"])
    s.add_dependency(%q<pry>, ["~> 0.10"])
    s.add_dependency(%q<rspec>, ["~> 3.0"])
    s.add_dependency(%q<rubocop>, ["~> 0.52"])
    s.add_dependency(%q<webmock>, ["~> 1.21"])
  end
end
